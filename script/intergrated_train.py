import sys
import os
import argparse
from math import prod
sys.path.append('../..')
# sys.path.append('/home/holywater2/2023/_Reproduce')

import repo_utils.debbug_utils as deb


parser = argparse.ArgumentParser(description='Intergrated MP Reproduce')

# Project and Model Selection
parser.add_argument('--project',            type=str,   default="schnet_periodic") #Wandb project name
parser.add_argument('--model',               type=str,   default="SchNet") #"e_form", "gap pbe", "bulk modulus","shear modulus", "elastic anisotropy"

# Hyperparameters
parser.add_argument('--batch_size',         type=int,   default=64)
parser.add_argument('--learning_rate',      type=float, default=0.001)
parser.add_argument('--weight_decay',       type=float, default=0.0005)
parser.add_argument('--prop',               type=str,   default="e_form") #"e_form", "gap pbe", "bulk modulus","shear modulus", "elastic anisotropy"

parser.add_argument('--n_epochs',           type=int,   default=500) # default 300
parser.add_argument('--radial_cutoff',      type=float, default=5.0) # default 5.0 (Graph construction할 때 cutoff)
parser.add_argument('--max_neighbors',      type=int,   default=12) # default 300

parser.add_argument('--n_layers',           type=int,     default=6)
parser.add_argument('--distance_cutoff',    type=float,   default=5.0) # SchNet default 5.0 (SchNet의 rbf의 cutoff)
parser.add_argument('--per_atom',           type=bool,    default=False) # Loss to per atom (Not good)
# parser.add_argument('--supercell_dim',      type=list,    default=None)
parser.add_argument('--supercell_dim', nargs='+', help='<Required> Set flag', default=None)


parser.add_argument('--scheduler',          type=str,   default="onecycle") #"onecycle", "None", "step"

# Dataset
parser.add_argument('--dataset',            type=str,   default="megnet")
parser.add_argument('--data_pdirname',      type=str,   default=None)
parser.add_argument('--data_dirname',       type=str,   default=None)

parser.add_argument('--loader_dirname',     type=str,   default=None)
parser.add_argument('--save_loader',        type=bool,  default=False)

parser.add_argument('--random_seed',        type=int,   default=None)  # default random

parser.add_argument('--n_train',            type=int,   default=None)  # default 60000
parser.add_argument('--n_val',              type=int,   default=None)  # default 5000
parser.add_argument('--n_test',             type=int,   default=None)  # default 4239

# Settings
parser.add_argument('--num_workers',        type=int,   default=8)
parser.add_argument('--metric_name',        type=str,   default="mae")
parser.add_argument('--mode',               type=str,   default=None)

# System argument
parser.add_argument('--GPU',                type=str,   default=None)  # default(None) is using all GPU, if want to use CPU, denote cpu
parser.add_argument('--det',                type=str,   default=None)  # For alignn reproduce
parser.add_argument('--wandb_disabled',     type=str,   default=False)
parser.add_argument('--num_threads',        type=int,   default=8)

# args = parser.parse_args()
args = parser.parse_args().__dict__


if args['GPU'] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]= args['GPU']

if args['det'] is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

deb.print_available_gpu()

config = {}

if args['supercell_dim'] is None:
        args['supercell_dim'] = [1,1,1]
else:
        args['supercell_dim'] = list(map(int, args['supercell_dim']))

args["num_atom_mul"] = prod(args['supercell_dim'])

for arg in args.keys():
        config[arg] = args[arg]
args['n_samples'] = None
        
# from models.train_schnet import SchNetPredictor
from dgllife.model import SchNetPredictor
from models.schnet_periodic import SchNetPeriodicPredictor
from models import train_schnet_like , train_alignn

if config['model'] == 'SchNet':
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
        
        train_schnet_like.train(args,config, model)

elif config['model'] == 'SchNet_Supercell':
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
        
        train_schnet_like.train(args,config, model)

elif config['model'] == 'SchNet_Supercell_PBC':
        layers = []
        for i in range(args['n_layers']):
                layers.append(64)
        
        model = SchNetPeriodicPredictor(node_feats=64,
                            hidden_feats=layers,
                            classifier_hidden_feats=64,
                            n_tasks=1,
                            num_node_types=100,
                            cutoff=config['distance_cutoff'],
                            gap=0.1,
                            predictor_hidden_feats=64,
                            max_neighbor=config['max_neighbors'])
        
        train_schnet_like.train(args,config, model)

elif config['model'] == 'Alignn':
        train_alignn.train(args,config)