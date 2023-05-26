# https://github.com/awslabs/dgl-lifesci/blob/master/examples/property_prediction/alchemy/main.py

import os
import sys
import wandb
import datetime
sys.path.append('..')
sys.path.append('/home/holywater2/2023/_Reproduce')


from repo_utils.data_utils import mp, dumpjson
from models.model_utils import count_parameters
# import repo_utils.debbug_utils as deb
from models.schnet_utils import *

from dgllife.model import SchNetPredictor
from alignn.data import get_train_val_loaders


def train(args, config):
    wandb_mode = None
    if args['wandb_disabled']:
        wandb_mode = "disabled"
        
    wandb.init(project=args["project"], 
            config=config,
            mode=wandb_mode,
            allow_val_change=True)

    for key in wandb.config.keys():
        args[key] = config[key]
    
    inp_config = config
    config = wandb.config
    
    if args['data_pdirname'] is None:
        data_pdirname = "../dataset/mp_megnet"
    else:
        data_pdirname = args['data_pdirname']
    
    datestr = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    output_dir = './results/' + config['model'] + '_' \
                + config['dataset'] + '_' \
                + config['prop'] + '_' \
                + datestr
    if args["mode"] == "sample":
        output_dir = output_dir + "_sample"
    print("[I] Output directory is",output_dir)
    os.makedirs(output_dir)
    
    dataset = []

    if args["mode"] == "sample":
        if args['loader_dirname'] is None:
            loader_dir  = data_pdirname+'/loader_sample'
        else:
            loader_dir  = args['loader_dirname']
        # if not config['save_loader']:
        if config['data_dirname'] is None:
            dataset = mp.load_json_sample(pdirname=data_pdirname)
        else :
            dataset = mp.load(pdirname=data_pdirname,
                                dirname=config['data_dirname'])
        train_loader, val_loader, test_loader, _ =\
            get_train_val_loaders(dataset="megnet",
                                dataset_array=dataset,
                                target=config["prop"],
                                batch_size=config["batch_size"],
                                filename=loader_dir,
                                save_dataloader=config['save_loader'],
                                output_dir=output_dir,
                                line_graph=False,
                                max_neighbors=config["max_neighbors"],
                                workers=config["num_workers"],
                                split_seed=config["random_seed"],
                                cutoff=config["radial_cutoff"],
                                # atom_features=,
                                id_tag="id")

    else:
        if args['loader_dirname'] is None:
            loader_dir  = data_pdirname+'/loader'
        else:
            loader_dir  = args['loader_dirname']
        # if not config['save_loader']:
        if config['data_dirname'] is None:
            dataset = mp.load(pdirname=data_pdirname)
        else :
            dataset = mp.load(pdirname=data_pdirname,
                                dirname=config['data_dirname'])
        train_loader, val_loader, test_loader, _ =\
            get_train_val_loaders(dataset="megnet",
                                dataset_array=dataset,
                                target=config["prop"],
                                batch_size=config["batch_size"],
                                filename=loader_dir,
                                save_dataloader=config['save_loader'],
                                output_dir=output_dir,
                                line_graph=False,
                                n_train=config["n_train"],
                                n_val=config["n_val"],
                                n_test=config["n_test"],
                                max_neighbors=config["max_neighbors"],
                                workers=config["num_workers"],
                                split_seed=config["random_seed"],
                                cutoff=config["radial_cutoff"],
                                id_tag="id")
    wandb.config.update({'n_train':len(train_loader.dataset),
                         'n_val':len(val_loader.dataset),
                         'n_test':len(test_loader.dataset)}
                        ,allow_val_change=True)
    # if config['n_train'] is None:
    #     config['n_train']   = len(train_loader.dataset)
    # if config['n_val'] is None:
    #     config['n_val']     = len(val_loader.dataset)
    # if config['n_test'] is None:
    #     config['n_test']    = len(test_loader.dataset)
    config['n_samples'] = config['n_test'] + config['n_val'] + config['n_train']

    print(config)
    dumpjson(inp_config,
             filename=output_dir+"/wandb_config.json")

    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # config['device'] = args['device']

    layers = []
    for i in range(args['n_layers']):
        layers.append(64)
    
    model = SchNetPredictor(node_feats=64,
                            hidden_feats=layers,
                            classifier_hidden_feats=64,
                            n_tasks=1,
                            num_node_types=100,
                            cutoff=config['distance_cutoff'],
                            gap=0.1,
                            predictor_hidden_feats=64)

    loss_fn = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
    
    print(config['n_epochs'],type(config['n_epochs']))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=config["learning_rate"],
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=config['n_epochs'])

    # if config.scheduler == "none":
    #     # always return multiplier of 1 (i.e. do nothing)
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(
    #         optimizer, lambda epoch: 1.0
    #     )

    # elif config.scheduler == "onecycle":
    #     steps_per_epoch = len(train_loader)
    #     # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer,
    #         max_lr=config.learning_rate,
    #         epochs=config.epochs,
    #         steps_per_epoch=steps_per_epoch,
    #         # pct_start=pct_start,
    #         pct_start=0.3,
    #     )
    # elif config.scheduler == "step":
    #     # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
    #     scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer,
    #     )

    # stopper = EarlyStopping(mode="lower", patience=10)

    model.to(args["device"])
    total_param = count_parameters(model)
    wandb.log({"total params":total_param},commit=False)

    for epoch in range(config["n_epochs"]):
        wandb.log({"cur_learning_rate":optimizer.param_groups[0]["lr"],
            "cur_weight_decay":optimizer.param_groups[0]["weight_decay"]},
            commit=False)           
        train_score, train_score_per_atom, train_loss = run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer,scheduler)
        print('epoch {:d}/{:d}, train {} {:.4f}, train {}/atom {:.4f}, , train loss {:.4f}'.format(
            epoch + 1, args["n_epochs"],
            args["metric_name"], train_score,
            args["metric_name"], train_score_per_atom,
            train_loss))
        
        wandb.log({"train/mae(eV)": train_score,
                   "train/mae_per_atom(eV/atom)":train_score_per_atom,
                   "train/loss": train_loss},
                  commit=False)


        
        # Validation and early stop
        val_score, val_score_per_atom ,val_loss = run_an_eval_epoch(args, model, val_loader, loss_fn)
        print('epoch {:d}/{:d}, valid {} {:.4f}, valid {}/atom {:.4f}, , valid loss {:.4f}'.format(
            epoch + 1, args["n_epochs"],
            args["metric_name"], val_score,
            args["metric_name"], val_score_per_atom,
            val_loss))
        
        wandb.log({"validation/mae(eV)":val_score,
                   "validation/mae_per_atom(eV/atom)":val_score_per_atom,
                   "validation/loss":val_loss},
                  commit=False)
                
        
        # test_score, test_score_per_atom, test_loss = run_an_eval_epoch(args, model, test_loader, loss_fn)
        # print('epoch {:d}/{:d}, test  {} {:.4f}, test  {}/atom {:.4f}, , test  loss {:.4f}'.format(
        #     epoch + 1, args["n_epochs"],
        #     args["metric_name"], test_score,
        #     args["metric_name"], test_score_per_atom,
        #     test_loss))
        
        # wandb.log({"test/mae(eV)":test_score,
        #            "test/mae_per_atom(eV/atom)":test_score_per_atom,
        #            "test/loss":test_loss},
        #           commit = False)
        
        wandb.log({"global_step":epoch+1})
        
    save_result = True
    run_an_fianl_eval(args, model, train_loader,loss_fn,filename=output_dir+"/train_",save=save_result)
    run_an_fianl_eval(args, model, val_loader,loss_fn,filename=output_dir+"/val_",save=save_result)
    test_score, test_score_per_atom, test_loss = \
        run_an_fianl_eval(args, model, test_loader,loss_fn, filename=output_dir+"/test_",save=save_result)
    
    print('epoch {:d}/{:d}, test  {} {:.4f}, test  {}/atom {:.4f}, , test  loss {:.4f}'.format(
        epoch + 1, args["n_epochs"],
        args["metric_name"], test_score,
        args["metric_name"], test_score_per_atom,
        test_loss))
    wandb.log({"final_test/mae(eV)":test_score,
               "final_test/mae_per_atom(eV/atom)":test_score_per_atom,
               "final_test/loss":test_loss})
    
    wandb.finish()