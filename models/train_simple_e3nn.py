
import os
import sys
import wandb
import datetime


sys.path.append('..')
sys.path.append('/home/holywater2/2023/_Reproduce')

import torch
import torch.nn as nn

from repo_utils.data_utils import mp
import repo_utils.debbug_utils as deb
from models.simple_e3nn import *

from repo_utils.data_utils import mp, dumpjson
from models.model_utils import count_parameters

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

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
    print("[I] Output directory is",output_dir)
    os.makedirs(output_dir)
    
    dataset = []
    if args["mode"] == "sample":
        if args['loader_dirname'] is None:
            loader_dir  = data_pdirname+'/loader_sample'
    else:
        if args['loader_dirname'] is None:
            loader_dir  = data_pdirname+'/loader'
    if args['loader_dirname'] is not None:
        loader_dir  = args['loader_dirname']
    
    if config['data_dirname'] is None:
        if args["mode"] == "sample":
            dataset = mp.load_json_sample(pdirname=data_pdirname)
        else:
            dataset = mp.load(pdirname=data_pdirname)
    else :
        dataset = mp.load(pdirname=data_pdirname,
                            dirname=config['data_dirname'])
        
    crystals, type_onehot, type_encoding = load_atoms_and_type_encoding(dataset)
    
    dataset = crystals_to_dataset(crystals,
                                type_onehot,
                                type_encoding,
                                args["radial_cutoff"],
                                save=True,
                                filename=output_dir)

    # Create data loaders for each split
    train_loader, val_loader, test_loader = split_dataset(dataset)

    config['n_train']   = len(train_loader.dataset)
    config['n_val']     = len(val_loader.dataset)
    config['n_test']    = len(test_loader.dataset)
    config['n_samples'] = config['n_test'] + config['n_val'] + config['n_train']

    if len(type_encoding)%2 == 0:
        irreps_in = str(len(type_encoding)+1) + "x0e"
    else:
        irreps_in = str(len(type_encoding)) + "x0e"


    model = SimplePeriodicNetwork(
        irreps_in= irreps_in,  # One hot scalars (L=0 and even parity) on each atom to represent atom type
        irreps_out="1x0e",  # Single scalar (L=0 and even parity) to output (for example) energy
        max_radius=args["radial_cutoff"], # Cutoff radius for convolution
        num_neighbors=args["num_neighbor"],  # scaling factor based on the typical number of neighbors
        pool_nodes=True,  # We pool nodes to predict total energy
    ).to(args["device"])



    loss_fn = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args["learning_rate"],
                                weight_decay=args["weight_decay"])

    for epoch in range(args["n_epochs"]):
        wandb.log({"global_step":epoch+1})
        
        train_score, train_loss = \
            run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)
        
        wandb.log({"train/mae": train_score , "train/loss": train_loss})
        
        val_score, val_loss = run_an_eval_epoch(args, model, val_loader,loss_fn)
        print('epoch {:d}/{:d}, validation {} {:.4f}'.format(
            epoch + 1, args["n_epochs"], args["metric_name"], val_score))
        wandb.log({"validation/mae":val_score, "validation/loss":val_loss})
        
        
    test_score, test_loss = run_an_eval_epoch(args, model, train_loader,loss_fn)
    print('test {} {:.4f}, test loss {:.4f}'.format(
        args["metric_name"], test_score, test_loss))

    wandb.log({"test/mae":test_score, "test/loss":test_loss})
    wandb.finish()
