
import os
import sys
import argparse
import wandb

sys.path.append('..')
sys.path.append('/home/holywater2/2023/_Reproduce')

import torch
import torch.nn as nn

from repo_utils.data_utils import mp
import repo_utils.debbug_utils as deb
from models.simple_e3nn import *

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

parser = argparse.ArgumentParser(description='simple e3nn MP Reproduce')

parser.add_argument('--project',            type=str,   default="simple_e3nn_periodic") #Wandb project name
parser.add_argument('--batch_size',         type=int,   default=48)
parser.add_argument('--learning_rate',      type=float, default=0.001)
parser.add_argument('--weight_decay',       type=float, default=0)
parser.add_argument('--dataset',            type=str,   default="megnet")
parser.add_argument('--data_pdirname',      type=str,   default=None)
parser.add_argument('--data_dirname',       type=str,   default=None)
parser.add_argument('--prop',               type=str,   default="e_form") #"e_form", "gap pbe", "bulk modulus","shear modulus", "elastic anisotropy"
parser.add_argument('--random_seed',        type=int,   default=123)  # default random
# parser.add_argument('--n_train',            type=int,   default=None)  # default 60000
# parser.add_argument('--n_val',              type=int,   default=None)  # default 5000
# parser.add_argument('--n_test',             type=int,   default=None)  # default 4239
parser.add_argument('--n_epochs',           type=int,   default=500) # default 300
parser.add_argument('--max_neighbors',      type=int,   default=12) # default 300
parser.add_argument('--num_neighbor',      type=int,   default=10) # default 300
parser.add_argument('--num_workers',        type=int,   default=0)
parser.add_argument('--radial_cutoff',      type=float, default=3.5)
parser.add_argument('--metric_name',        type=str,   default="mae")
parser.add_argument('--mode',               type=str,   default="sample")
# parser.add_argument('--load_dataset', type=bool, default=False)


# System argument
parser.add_argument('--GPU',                type=str,   default=None)  # default(None) is using all GPU, if want to use CPU, denote cpu
parser.add_argument('--det',                type=str,   default=None)

# args = parser.parse_args()
args = parser.parse_args().__dict__


config={
        "learning_rate"  :args["learning_rate"],
        "dataset"        :args["dataset"],
        "batch_size"     :args["batch_size"],
        "weight_decay"   :args["weight_decay"],
        "prop"           :args["prop"],
        "random_seed"    :args["random_seed"],
        "num_workers"    :args["num_workers"],
        "n_epochs"       :args["n_epochs"],
        "max_neighbors"  :args["max_neighbors"],
        "mode"           :args["mode"],
        "radial_cutoff"  :args["radial_cutoff"],
        "num_neighbor"   :args["num_neighbor"]
        # "n_train"        :args.n_train,
        # "n_val"          :args.n_val,
        # "n_test"         :args.n_test
        }


# if not args["load_dataset"] :
if args["mode"] == "sample":
    dataset = mp.load_json_sample()
else:
    dataset = mp.load()

config['n_samples'] = len(dataset)

wandb.init(project=args["project"], 
           config=config)

for key in wandb.config.keys():
    args[key] = config[key]

print(config)

args["device"] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
deb.print_available_gpu()


crystals, type_onehot, type_encoding = load_atoms_and_type_encoding(dataset)

if args["mode"] == "sample":
    save_data_name = "e3nn_mp_sample"
else:
    save_data_name = "e3nn_mp"
    
dataset = crystals_to_dataset(crystals,
                              type_onehot,
                              type_encoding,
                              args["radial_cutoff"],
                              save=True,
                              filename=save_data_name)



# Create data loaders for each split
train_loader, val_loader, test_loader = split_dataset(dataset)

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