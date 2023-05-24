import e3nn

import torch
import numpy as np
import os
import torch_geometric
import torch_geometric.data
import torch_scatter
from torch.utils.data.dataset import random_split
from typing import Dict, Union
from tqdm import tqdm

from jarvis.core.atoms import Atoms
import ase
import ase.neighborlist

from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
from dgllife.utils import EarlyStopping, Meter

tqdm.pandas()

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


def load_atoms_and_type_encoding(dataset):
    print("Generationg ase atoms and type encoding...")
    type_encoding = {}
    crystals = []
    num_atom_types = 0
    for crystal_dict in tqdm(dataset):
        target = crystal_dict["e_form"]
        crystal = Atoms.from_dict(crystal_dict["atoms"]).ase_converter()
        for atom in crystal.symbols:
            if atom not in type_encoding:
                type_encoding[atom] = num_atom_types
                num_atom_types += 1
        crystals.append([crystal,target])

    type_onehot = torch.eye(len(type_encoding))
    print("Generationg Completed! The number of the atom type is",num_atom_types)
    
    return crystals, type_onehot, type_encoding

def crystals_to_dataset(crystals, type_onehot, type_encoding, radial_cutoff,
                        save=False, dirname="../dataset/simple_e3nn_mp",
                        filename="mp_e3nn"):
    filename = filename + '.data'
    if save:
        if not os.path.exists(dirname):
            print("Warning: can not find", os.getcwd()+'/'+ dirname)
            print("It will cause save error!")
            return
        else:
            os.chdir(dirname)
            if not os.path.exists(filename):
                print("Warning: can not find", os.getcwd()+'/'+ filename)
                print(filename,"will be generated after transforming!")
            else:
                print("Loading dataset...")
                dataset = torch.load(filename)
                print("Loading dataset is completed!")
                return dataset
                
    print("Transforming ase atoms into dataset format...")
    dataset = []
    for crystal, target in tqdm(crystals):
    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images / copies of the unit cell
        edge_src, edge_dst, edge_shift = \
            ase.neighborlist.neighbor_list("ijS",
                                           a=crystal,
                                           cutoff=radial_cutoff,
                                           self_interaction=True)

        data = torch_geometric.data.Data(
            pos=torch.tensor(crystal.get_positions()),
            lattice=torch.tensor(crystal.cell.array).unsqueeze(0),  # We add a dimension for batching
            x=type_onehot[[type_encoding[atom] for atom in crystal.symbols]],  # Using "dummy" inputs of scalars because they are all C
            edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
            edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
            energy=target  # dummy energy (assumed to be normalized "per atom")
        )
        dataset.append(data)
        
    print("Transforming Complted!")

    if save:
        torch.save(dataset,filename)
        
    return dataset

def split_dataset(dataset,
                  train_ratio = 0.8,
                  val_ratio   = 0.1,
                  test_ratio  = 0.1,
                  batch_size  = 2,
                  train_shuffle =True):
    
    # Calculate the lengths of each split
    num_samples = len(dataset)
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size

    # Split the dataset into train, validation, and test sets
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders for each split
    train_loader = torch_geometric.loader.DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle)
    val_loader   = torch_geometric.loader.DataLoader(val_set  , batch_size=batch_size)
    test_loader  = torch_geometric.loader.DataLoader(test_set , batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


class SimplePeriodicNetwork(SimpleNetwork):
    def __init__(self, **kwargs):
        """The keyword `pool_nodes` is used by SimpleNetwork to determine
        whether we sum over all atom contributions per example. In this example,
        we want use a mean operations instead, so we will override this behavior.
        """
        self.pool = False
        if kwargs['pool_nodes'] == True:
            kwargs['pool_nodes'] = False
            kwargs['num_nodes'] = 1.
            self.pool = True
        super().__init__(**kwargs)

    # Overwriting preprocess method of SimpleNetwork to adapt for periodic boundary data
    def preprocess(self, data: Union[torch_geometric.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        edge_src = data['edge_index'][0]  # Edge source
        edge_dst = data['edge_index'][1]  # Edge destination

        # We need to compute this in the computation graph to backprop to positions
        # We are computing the relative distances + unit cell shifts from periodic boundaries
        edge_batch = batch[edge_src]
        edge_vec = (data['pos'][edge_dst]
                    - data['pos'][edge_src]
                    + torch.einsum('ni,nij->nj', data['edge_shift'], data['lattice'][edge_batch]))

        return batch, data['x'], edge_src, edge_dst, edge_vec

    def forward(self, data: Union[torch_geometric.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        # if pool_nodes was set to True, use scatter_mean to aggregate
        output = super().forward(data)
        if self.pool == True:
            return torch_scatter.scatter_mean(output, data.batch, dim=0)  # Take mean over atoms per example
        else:
            return output

def run_a_train_epoch(args,epoch, model, data_loader,
                      loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in tqdm(enumerate(data_loader)):
        data, labels = batch_data, batch_data["energy"]
        data = data.to(args["device"])
        labels = labels.reshape([-1,1])
        labels = labels.to(args["device"])
        
        prediction = model(data)
        loss = (loss_criterion(prediction, labels)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_meter.update(prediction, labels)
        
    total_score = np.mean(train_meter.compute_metric(args["metric_name"]))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args["n_epochs"], args["metric_name"], total_score))
    return total_score, loss
    
def run_an_eval_epoch(args, model, data_loader, loss_criterion):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in tqdm(enumerate(data_loader)):
            data, labels = batch_data, batch_data["energy"]
            data = data.to(args["device"])
            labels = labels.reshape([-1,1])
            labels = labels.to(args["device"])
            
            prediction = model(data)
            eval_loss = (loss_criterion(prediction, labels)).mean()
            eval_meter.update(prediction, labels)
        
        total_score = np.mean(eval_meter.compute_metric(args["metric_name"]))
    return total_score, eval_loss