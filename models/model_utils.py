import torch
import torch.nn as nn
from prettytable import PrettyTable

def gaussian_basis(
    distance, max_distance, num_centers, width, min_distance=0, to_device=True
):
    '''
    creates soft binning of distance
    '''
    centers = torch.linspace(min_distance, max_distance, num_centers)
    
    if to_device:
        centers = centers.to(distance.device)
    
    # in case identifier for virutal edges is present
    positions = centers - distance
    gaussian_expansion = torch.exp(-positions * positions / (width * width))
    
    return gaussian_expansion


class EmbeddingLayer(nn.Module):
    
    def __init__(self, in_features, emb_size):
        
        super().__init__()
        
        self.emb = nn.Linear(in_features, emb_size)
        
    def forward(self, x):
        
        return self.emb(x)

def count_parameters_def(model):
    return sum(p.numel() for p in model.parameters())

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
