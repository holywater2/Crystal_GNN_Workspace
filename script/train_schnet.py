# https://github.com/awslabs/dgl-lifesci/blob/master/examples/property_prediction/alchemy/main.py

import os
import sys
import argparse
import wandb
sys.path.append('..')
sys.path.append('/home/holywater2/2023/_Reproduce')

import numpy as np
import torch
import torch.nn as nn

from repo_utils.data_utils import mp
import repo_utils.debbug_utils as deb

from dgllife.model import SchNetPredictor
from dgllife.utils import EarlyStopping, Meter

from alignn.data import get_train_val_loaders

parser = argparse.ArgumentParser(description='SchNet MP Reproduce')

parser.add_argument('--project',            type=str,   default="schnet_periodic") #Wandb project name
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
parser.add_argument('--num_workers',        type=int,   default=0)
parser.add_argument('--metric_name',        type=str,   default="mae")
parser.add_argument('--mode',               type=str,   default="sample")



# System argument
parser.add_argument('--GPU',                type=str,   default=None)  # default(None) is using all GPU, if want to use CPU, denote cpu
parser.add_argument('--det',                type=str,   default=None)

# args = parser.parse_args()
args = parser.parse_args().__dict__


# dataset = mp.load_json(pdirname=args.data_pdirname,
#                        dirname=args.data_dirname,
#                        down=False)

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
        "mode"           :args["mode"]
        # "n_train"        :args.n_train,
        # "n_val"          :args.n_val,
        # "n_test"         :args.n_test
        }



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


if args["mode"] == "sample":
    train_loader, val_loader, test_loader, _ =\
        get_train_val_loaders(dataset="megnet",
                            dataset_array=dataset,
                            target=args["prop"],
                            batch_size=args["batch_size"],
                            filename="sample",
                            save_dataloader=True,
                            output_dir="sample_out",
                            line_graph=False,
                            max_neighbors=args["max_neighbors"],
                            workers=args["num_workers"],
                            split_seed=args["random_seed"],
                            id_tag="id")

else :    
    train_loader, val_loader, test_loader, _ =\
        get_train_val_loaders(dataset="megnet",
                            dataset_array=dataset,
                            target=args["prop"],
                            batch_size=args["batch_size"],
                            filename="sample_sweep",
                            save_dataloader=True,
                            output_dir="sample_out",
                            line_graph=False,
                            max_neighbors=args["max_neighbors"],
                            workers=args["num_workers"],
                            split_seed=args["random_seed"],
                            id_tag="id")
    
metric = args["metric_name"]

def regress(args, model, bg):
    bg = bg.to(args["device"])
    node_types      = bg.ndata.pop('atomic_number').type('torch.LongTensor')
    edge_distances  = bg.edata.pop('r').norm(dim=1).reshape(-1,1)
    node_types      = node_types.to(args["device"])
    edge_distances  = edge_distances.to(args["device"])
    return model(bg, node_types, edge_distances)

def run_a_train_epoch(args,epoch, model, data_loader,
                      loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        bg, labels = batch_data
        labels = labels.reshape([-1,1,1])
        labels = labels.to(args["device"])
        prediction = regress(args, model, bg)
        loss = (loss_criterion(prediction, labels)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels)
    total_score = np.mean(train_meter.compute_metric(metric))
    wandb.log({"train/mae": total_score , "train/loss": loss})
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args["n_epochs"], metric, total_score))

def run_an_eval_epoch(args, model, data_loader, loss_criterion):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            bg, labels = batch_data
            labels = labels.reshape([-1,1,1])
            labels = labels.to(args["device"])
            prediction = regress(args, model, bg)
            eval_loss = (loss_criterion(prediction, labels)).mean()
            eval_meter.update(prediction, labels)
        total_score = np.mean(eval_meter.compute_metric(metric))
    return total_score, eval_loss

model = SchNetPredictor(node_feats=64,
                        hidden_feats=None,
                        classifier_hidden_feats=64,
                        n_tasks=1,
                        num_node_types=100,
                        cutoff=30.,
                        gap=0.1,
                        predictor_hidden_feats=64)

loss_fn = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args["learning_rate"],
                             weight_decay=args["weight_decay"])

stopper = EarlyStopping(mode="lower", patience=10)

model.to(args["device"])

for epoch in range(args["n_epochs"]):
    wandb.log({"global_step":epoch+1})
        
    run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)

    # Validation and early stop
    val_score, val_loss = run_an_eval_epoch(args, model, val_loader,loss_fn)
    # early_stop = stopper.step(val_score, model)
    print('epoch {:d}/{:d}, validation {} {:.4f}'.format(
        epoch + 1, args["n_epochs"], args["metric_name"], val_score))
    wandb.log({"validation/mae":val_score, "validation/loss":val_loss})

    # if early_stop:
    #     break
    
test_score, test_loss = run_an_eval_epoch(args, model, train_loader,loss_fn)
print('test {} {:.4f}, test loss {:.4f}'.format(
    metric, test_score, test_loss))

wandb.log({"test/mae":test_score, "test/loss":test_loss})
wandb.finish()