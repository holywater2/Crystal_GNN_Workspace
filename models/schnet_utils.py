import numpy as np
import os
import torch
import torch.nn as nn
import csv
from dgllife.utils import EarlyStopping, Meter
from datetime import date

"""Torch modules for interaction blocks in SchNet"""
# pylint: disable= no-member, arguments-differ, invalid-name
import dgl.function as fn

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
                      scheduler,
                      save = False, per_atom = False, filename = ""):
    model.train()
    train_meter = Meter()
    train_meter_per_atom = Meter()
    epoch_loss = 0.0
    
    for batch_id, batch_data in tqdm(enumerate(data_loader)):
        bg, labels = batch_data
        labels = labels.reshape([-1,1])
        labels = labels.to(args["device"])
        prediction = regress(args, model, bg)
        num_atom = (bg.batch_num_nodes().reshape([-1,1])/args["num_atom_mul"]).to(args["device"])
        
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
    
    save_data_list = []    
    if save:
        filename1 = filename + 'result.csv'
        savedata = torch.hstack([torch.vstack(train_meter.y_true), torch.vstack(train_meter.y_pred)]).cpu().numpy()
        
        save_data_list.append([filename1,savedata, total_score, epoch_loss])
        
        if per_atom:
            filename2 = filename + 'result_per_atom.csv'
            savedata_pa = torch.hstack([torch.vstack(train_meter_per_atom.y_true), torch.vstack(train_meter_per_atom.y_pred)]).cpu().numpy()
            save_data_list.append([filename2, savedata_pa, total_score_per_atom, epoch_loss])

    return total_score, total_score_per_atom, epoch_loss, save_data_list

def run_an_eval_epoch(args, model, data_loader, loss_criterion,
                      save = False, per_atom = False, filename = ""):
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
            
            num_atom = (bg.batch_num_nodes().reshape([-1,1])/args["num_atom_mul"]).to(args["device"])
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
    
    save_data_list = []    
    if save:
        filename1 = filename + 'result.csv'
        savedata = torch.hstack([torch.vstack(eval_meter.y_true), torch.vstack(eval_meter.y_pred)]).cpu().numpy()
        
        save_data_list.append([filename1,savedata, total_score, epoch_loss])
        
        if per_atom:
            filename2 = filename + 'result_per_atom.csv'
            savedata_pa = torch.hstack([torch.vstack(eval_meter_per_atom.y_true), torch.vstack(eval_meter_per_atom.y_pred)]).cpu().numpy()
            save_data_list.append([filename2, savedata_pa, total_score_per_atom, epoch_loss])

    return total_score, total_score_per_atom, epoch_loss, save_data_list

def save_csv_data(args, save_data_list):
    
    for filename, savedata, total_score, epoch_loss in save_data_list:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([args["metric_name"],"loss"])
            writer.writerow([total_score, epoch_loss])
            writer.writerow(["y_true","y_pred"])
            for row in savedata:
                writer.writerow(row)