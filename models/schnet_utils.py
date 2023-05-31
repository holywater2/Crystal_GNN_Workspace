import numpy as np
import os
import torch
import torch.nn as nn
import csv
from dgllife.utils import EarlyStopping, Meter
from datetime import date

from tqdm import tqdm

from ignite.metrics import Loss, MeanAbsoluteError
# MeanAbsoluteError()

type_onehot = torch.eye(100).type('torch.LongTensor')

def regress(args, model, bg):
    node_types      = bg.ndata.pop('atomic_number').reshape(-1).type('torch.LongTensor')
    node_types      = node_types.to(args["device"])
    
    # atomic_numbers  = bg.ndata.pop('atomic_number').int().reshape(-1).tolist()
    # node_types      = type_onehot[atomic_numbers]
    
    edge_distances  = bg.edata.pop('r').norm(dim=1).reshape(-1,1)
    edge_distances  = edge_distances.to(args["device"])
 
    bg = bg.to(args["device"])
    
    return model(bg, node_types, edge_distances)

def run_a_train_epoch(args,epoch, model, data_loader,
                      loss_criterion, optimizer,
                      scheduler):
    model.train()
    train_meter = Meter()
    train_meter_per_atom = Meter()
    epoch_loss = 0.0
    
    for batch_id, batch_data in tqdm(enumerate(data_loader)):
        bg, labels = batch_data
        labels = labels.reshape([-1,1])
        labels = labels.to(args["device"])
        prediction = regress(args, model, bg)
        num_atom = bg.batch_num_nodes().reshape([-1,1]).to(args["device"])
        
        if args["per_atom"]:
            loss = (loss_criterion(prediction/num_atom, labels/num_atom)).mean()
        else:
            loss = (loss_criterion(prediction, labels)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_meter.update(prediction, labels)
        train_meter_per_atom.update(prediction/num_atom, labels/num_atom)
        epoch_loss += loss.item() * bg.batch_size
        
    epoch_loss /= len(data_loader.dataset)
    total_score = np.mean(train_meter.compute_metric(args["metric_name"]))
    total_score_per_atom = np.mean(train_meter_per_atom.compute_metric(args["metric_name"]))

    return total_score, total_score_per_atom, epoch_loss

def run_an_eval_epoch(args, model, data_loader, loss_criterion):
    model.eval()
    eval_meter = Meter()
    eval_meter_per_atom = Meter()
    epoch_loss = 0.0
    
    with torch.no_grad():
        for batch_id, batch_data in tqdm(enumerate(data_loader)):
            bg, labels = batch_data
            labels = labels.reshape([-1,1])
            labels = labels.to(args["device"])
            prediction = regress(args, model, bg)
            num_atom = bg.batch_num_nodes().reshape([-1,1]).to(args["device"])
            if args["per_atom"]:
                eval_loss = (loss_criterion(prediction/num_atom, labels/num_atom)).mean()
            else:
                eval_loss = (loss_criterion(prediction, labels)).mean()

            eval_meter.update(prediction, labels)
            eval_meter_per_atom.update(prediction/num_atom, labels/num_atom)
            epoch_loss += eval_loss.item() * bg.batch_size
            
        epoch_loss /= len(data_loader.dataset)
        total_score = np.mean(eval_meter.compute_metric(args["metric_name"]))
        total_score_per_atom = np.mean(eval_meter_per_atom.compute_metric(args["metric_name"]))
        
    return total_score, total_score_per_atom, epoch_loss

def run_an_fianl_eval(args, model, data_loader, loss_criterion,
                      save = True, filename = ""):
    model.eval()
    eval_meter = Meter()
    eval_meter_per_atom = Meter()
    epoch_loss = 0.0
    
    with torch.no_grad():
        for batch_id, batch_data in tqdm(enumerate(data_loader)):
            bg, labels = batch_data
            labels = labels.reshape([-1,1])
            labels = labels.to(args["device"])
            
            prediction = regress(args, model, bg)
            
            num_atom = bg.batch_num_nodes().reshape([-1,1]).to(args["device"])
            if args["per_atom"]:
                eval_loss = (loss_criterion(prediction/num_atom, labels/num_atom)).mean()
            else:
                eval_loss = (loss_criterion(prediction, labels)).mean()

            eval_meter.update(prediction, labels)
            eval_meter_per_atom.update(prediction/num_atom, labels/num_atom)
            epoch_loss += eval_loss.item() * bg.batch_size
            
        epoch_loss /= len(data_loader.dataset)
        total_score = np.mean(eval_meter.compute_metric(args["metric_name"]))
        total_score_per_atom = np.mean(eval_meter_per_atom.compute_metric(args["metric_name"]))
        # print(prediction)
        # print(num_atom)
        # print(prediction/num_atom)        
    if save:
        filename1 = filename + 'result.csv'
        filename2 = filename + 'result_per_atom.csv'
        
        # print(torch.vstack(eval_meter.y_true).shape)
        # print([eval_meter.y_true, eval_meter.y_pred])
        savedata = torch.hstack([torch.vstack(eval_meter.y_true), torch.vstack(eval_meter.y_pred)]).cpu().numpy()
        savedata_pa = torch.hstack([torch.vstack(eval_meter_per_atom.y_true), torch.vstack(eval_meter_per_atom.y_pred)]).cpu().numpy()

        with open(filename1, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([args["metric_name"],"loss"])
            writer.writerow([total_score, epoch_loss])
            writer.writerow(["y_true","y_pred"])
            for row in savedata:
                writer.writerow(row)
        with open(filename2, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([args["metric_name"],"loss"])
            writer.writerow([total_score_per_atom, epoch_loss])
            writer.writerow(["y_true","y_pred"])
            for row in savedata_pa:
                writer.writerow(row)
    return total_score, total_score_per_atom, epoch_loss
    