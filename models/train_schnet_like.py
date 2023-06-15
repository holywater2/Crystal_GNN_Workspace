# https://github.com/awslabs/dgl-lifesci/blob/master/examples/property_prediction/alchemy/main.py

import sys
import wandb
sys.path.append('..')
sys.path.append('/home/holywater2/2023/_Reproduce')

from repo_utils.data_utils import mp, dumpjson, get_train_val_loaders
from models.model_utils import count_parameters
# import repo_utils.debbug_utils as deb
from models.schnet_utils import *

# from dgllife.model import SchNetPredictor
# from models.schnet_periodic_test import SchNetPredictor
from dgllife.model import SchNetPredictor

# from alignn.data import get_train_val_loaders

from models.train_utils import wandb_init_and_output_dir


def train(args, config, inp_model):
    torch.set_num_threads(args['num_threads'])
    
    args, config, inp_config = wandb_init_and_output_dir(args, config)
    
    dataset = []

    if args['loader_dirname'] is None:
        args['loader_dir']  = args['data_pdirname']+'/loader'
    else:
        args['loader_dir']  = args['loader_dirname']
    # if not config['save_loader']:
    if config['data_dirname'] is None:
        if args["mode"] == "sample":
            dataset = mp.load_json_sample(pdirname=args['data_pdirname'])
        else:
            dataset = mp.load(pdirname=args['data_pdirname'])            
    else :
        dataset = mp.load(pdirname=args['data_pdirname'],
                            dirname=config['data_dirname'])
    train_loader, val_loader, test_loader, _ =\
        get_train_val_loaders(dataset="megnet",
                            dataset_array=dataset,
                            target=config["prop"],
                            batch_size=config["batch_size"],
                            filename=args['loader_dir'],
                            save_dataloader=config['save_loader'],
                            output_dir=args['output_dir'],
                            line_graph=False,
                            max_neighbors=config["max_neighbors"],
                            workers=config["num_workers"],
                            split_seed=config["random_seed"],
                            cutoff=config["radial_cutoff"],
                            # atom_features=,
                            id_tag="id",
                            supercell_dim=args["supercell_dim"]
                            )
            
    wandb.config.update({'n_train':len(train_loader.dataset),
                         'n_val':len(val_loader.dataset),
                         'n_test':len(test_loader.dataset)}
                        ,allow_val_change=True)

    config['n_samples'] = config['n_test'] + config['n_val'] + config['n_train']

    print(config)
    dumpjson(inp_config,filename=args['output_dir']+"/wandb_config.json")

    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # config['device'] = args['device']

    """ Model initializing """    
    model = inp_model

    """ Loss and optimizer """
    loss_fn = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
    
    print(config['n_epochs'],type(config['n_epochs']))
    
    """ Scheduler """
    if config["scheduler"] == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config["scheduler"] == "onecycle":
        steps_per_epoch = len(train_loader)
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["learning_rate"],
            epochs=config["n_epochs"],
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    elif config["scheduler"] == "step":
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
        )

    # stopper = EarlyStopping(mode="lower", patience=10)
    best_val_loss = float('inf')
    best_test_score = None

    model.to(args["device"])
    total_param = count_parameters(model)
    wandb.log({"total params":total_param},commit=False)
    
    save_result = True
    per_atom = True

    for epoch in range(config["n_epochs"]):
        """ Learning rate log """
        wandb.log({"cur_learning_rate":optimizer.param_groups[0]["lr"],
            "cur_weight_decay":optimizer.param_groups[0]["weight_decay"]},
            commit=False)   
        
        """ Train metric log """
        train_score, train_score_per_atom, train_loss, train_save_data = run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer,scheduler,
                                                                                           save_result, per_atom, filename=args["output_dir"]+"/train_")
        print('epoch {:d}/{:d}, train {} {:.4f}, train {}/atom {:.4f}, , train loss {:.4f}'.format(
            epoch + 1, args["n_epochs"],
            args["metric_name"], train_score,
            args["metric_name"], train_score_per_atom,
            train_loss))
        
        wandb.log({"train/mae(eV)": train_score,
                   "train/mae_per_atom(eV_atom)":train_score_per_atom,
                   "train/loss": train_loss},
                  commit=False)
        
        """ Validation metric log """
        val_score, val_score_per_atom ,val_loss, val_save_data = run_an_eval_epoch(args, model, val_loader, loss_fn,
                                                                                   save_result, per_atom, filename=args["output_dir"]+"/val_")
        print('epoch {:d}/{:d}, valid {} {:.4f}, valid {}/atom {:.4f}, , valid loss {:.4f}'.format(
            epoch + 1, args["n_epochs"],
            args["metric_name"], val_score,
            args["metric_name"], val_score_per_atom,
            val_loss))
        
        wandb.log({"validation/mae(eV)":val_score,
                   "validation/mae_per_atom(eV_atom)":val_score_per_atom,
                   "validation/loss":val_loss},
                  commit=False)
        
        if val_loss < best_val_loss:
            best_train_save_data = train_save_data
            best_val_loss = val_loss
            best_val_save_data = val_save_data
            test_score, test_score_per_atom ,test_loss, test_save_data = run_an_eval_epoch(args, model, test_loader, loss_fn,
                                                                                           save_result, per_atom, filename=args["output_dir"]+"/test_")
            # best_test_score = test_score
        
            print('epoch {:d}/{:d}, test  {} {:.4f}, test  {}/atom {:.4f}, , test  loss {:.4f}'.format(
                epoch + 1, args["n_epochs"],
                args["metric_name"], test_score,
                args["metric_name"], test_score_per_atom,
                test_loss))
            
        # wandb.log({"test/mae(eV)":test_score,
        #            "test/mae_per_atom(eV/atom)":test_score_per_atom,
        #            "test/loss":test_loss},
        #           commit = False)
        
        wandb.log({"global_step":epoch+1})
    
    if save_result:
        save_csv_data(args,best_train_save_data)
        save_csv_data(args,best_val_save_data)
        save_csv_data(args,test_save_data)
    
    print('epoch {:d}/{:d}, test  {} {:.4f}, test  {}/atom {:.4f}, , test  loss {:.4f}'.format(
        epoch + 1, args["n_epochs"],
        args["metric_name"], test_score,
        args["metric_name"], test_score_per_atom,
        test_loss))
    wandb.log({"final_test/mae(eV)":test_score,
               "final_test/mae_per_atom(eV_atom)":test_score_per_atom,
               "final_test/loss":test_loss})
    
    wandb.finish()