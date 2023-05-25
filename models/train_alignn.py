# https://github.com/awslabs/dgl-lifesci/blob/master/examples/property_prediction/alchemy/main.py

import os
import sys
import wandb
import datetime
sys.path.append('..')
sys.path.append('/home/holywater2/2023/_Reproduce')

import torch

from repo_utils.data_utils import mp, dumpjson
from models.model_utils import count_parameters
from models.alignn_utils import *

from dgllife.model import SchNetPredictor
from alignn.data import get_train_val_loaders
# from alignn.train import train_dgl

def train(args, config):
    wandb_mode = None
    if args['wandb_disabled']:
        wandb_mode = "disabled"
        
    wandb.init(project=args["project"], 
            config=config,
            sync_tensorboard=True,
            mode=wandb_mode)

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
    
    loading_dataset = False
    if loading_dataset :
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
                                    line_graph=True,
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
                                    line_graph=True,
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
    # config['n_samples'] = config['n_test'] + config['n_val'] + config['n_train']

    print(config)
    dumpjson(inp_config,
             filename=output_dir+"/wandb_config.json")

    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # config['device'] = args['device']
    
    model_config={
        "learning_rate"  :args['learning_rate'],
        "dataset"        :args['dataset'],
        "batch_size"     :args['batch_size'],
        "prop"         :args['prop'],
        # "target"         :args['prop'],
        "random_seed"    :args['random_seed'],
        "num_workers"    :args['num_workers'],
        # "epochs"         :args['n_epochs'],
        "n_epochs"         :args['n_epochs'],
        "output_dir"     :output_dir,
        "weight_decay": args['weight_decay'],
        # "scheduler": scheduler,
        # "save_dataloader": save_dataloader,
        # "write_predictions": write_predictions,
        # "num_workers": num_workers,
        # "classification_threshold": classification_threshold,
        # "model": {"name": name, },
        # "log_tensorboard": log_tensorboard
        }
    
    train_prop_model(**model_config)
    # train_dgl(model_config,
    #           train_val_test_loaders= [train_loader, val_loader, test_loader, _ ])
 
    wandb.finish()