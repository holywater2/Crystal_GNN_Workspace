# from dgllife.model import SchNetPredictor
from dgllife.model.gnn.schnet import RBFExpansion, Interaction

import numpy as np
import torch
import torch.nn as nn

from dgl.nn.pytorch import CFConv
from dgl.nn.pytorch.conv.cfconv import ShiftedSoftplus
from dgllife.model.readout import MLPNodeReadout


import copy
import torch.nn.init as init

class TimeInteraction(nn.Module):
    """Building block for SchNet.

    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.

    This layer combines node and edge features in message passing and updates node
    representations.

    Parameters
    ----------
    node_feats : int
        Size for the input and output node features.
    edge_in_feats : int
        Size for the input edge features.
    hidden_feats : int
        Size for hidden representations.
    """
    def __init__(self, node_feats, time_step):
        super(TimeInteraction, self).__init__()
        
        # self.lin1 = nn.Linear(node_feats*time_step,node_feats*time_step)
        self.lin2 = nn.Sequential(
                nn.Linear(node_feats*time_step, 256),
                ShiftedSoftplus(),
                nn.Linear(256, node_feats*time_step)
            )

        
    def reset_parameters(self):
        """Reinitialize model parameters."""
        # self.lin1.reset_parameters()
        
        for module in self.lin2.modules():
            if isinstance(module, nn.Linear):
                # 가중치 초기화
                init.xavier_uniform_(module.weight)
                # 편향 초기화
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, node_feats_with_times):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feats)
            Input node features, V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (V, node_feats)
            Updated node representations.
        """
        return self.lin2(node_feats_with_times)
    
class TimeInteraction_NodeWise(nn.Module):
    """Building block for SchNet.

    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.

    This layer combines node and edge features in message passing and updates node
    representations.

    Parameters
    ----------
    node_feats : int
        Size for the input and output node features.
    edge_in_feats : int
        Size for the input edge features.
    hidden_feats : int
        Size for hidden representations.
    """
    def __init__(self, node_feats, time_step, hidden_feats=64):
        super(TimeInteraction_NodeWise, self).__init__()
        
        # self.lin1 = nn.Linear(node_feats*time_step,node_feats*time_step)
        self.lin2 = nn.Sequential(
                nn.Linear(time_step, hidden_feats),
                ShiftedSoftplus(),
                nn.Linear(hidden_feats, time_step)
            )

        
    def reset_parameters(self):
        """Reinitialize model parameters."""
        # self.lin1.reset_parameters()
        
        for module in self.lin2.modules():
            if isinstance(module, nn.Linear):
                # 가중치 초기화
                init.xavier_uniform_(module.weight)
                # 편향 초기화
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, node_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feats)
            Input node features, V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (V, node_feats)
            Updated node representations.
        """
        return self.lin2(node_feats)
    

class SchNetGNN_timestep(nn.Module):
    """SchNet.

    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.

    This class performs message passing in SchNet and returns the updated node representations.

    Parameters
    ----------
    node_feats : int
        Size for node representations to learn. Default to 64.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of hidden representations for the i-th interaction
        layer. ``len(hidden_feats)`` equals the number of interaction layers.
        Default to ``[64, 64, 64]``.
    num_node_types : int
        Number of node types to embed. Default to 100.
    cutoff : float
        Largest center in RBF expansion. Default to 30.
    gap : float
        Difference between two adjacent centers in RBF expansion. Default to 0.1.
    """
    def __init__(self, node_feats=64, hidden_feats=None, num_node_types=100, cutoff=30., gap=0.1,
                 timestep = 10):
        super(SchNetGNN_timestep, self).__init__()
        
        self.timestep = timestep

        if hidden_feats is None:
            hidden_feats = [64, 64, 64]

        self.embed = nn.Embedding(num_node_types, node_feats)
        self.rbf = RBFExpansion(high=cutoff, gap=gap)

        n_layers = len(hidden_feats)
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                Interaction(node_feats, len(self.rbf.centers), hidden_feats[i])
                )
        self.gnn_layers_timestep = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers_timestep.append(
                # TimeInteraction(node_feats,timestep)
                TimeInteraction_NodeWise(node_feats,timestep)
            )
        
        self.timereadout = nn.Sequential(
                nn.Linear(timestep, 32),
                ShiftedSoftplus(),
                nn.Linear(32, 1)
            )


        
    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.embed.reset_parameters()
        self.rbf.reset_parameters()
        for layer in self.gnn_layers:
            layer.reset_parameters()
        for layer in self.gnn_layers_timestep:
            layer.reset_parameters()

    def forward(self, g, node_types, edge_dists):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_types : int64 tensor of shape (V)
            Node types to embed, V for the number of nodes.
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.

        Returns
        -------
        node_feats : float32 tensor of shape (V, node_feats)
            Updated node representations.
        """
        node_feats = self.embed(node_types)
        expanded_dists = self.rbf(edge_dists)
        set_of_g = []
        for i in range(self.timestep):
            # set_of_g.append([copy.deepcopy(g),copy.deepcopy(node_feats),copy.deepcopy(expanded_dists)])
            set_of_g.append([g.clone(),node_feats.clone(),expanded_dists.clone()])

        # for gnn in self.gnn_layers:
        #     node_feats = gnn(g, node_feats, expanded_dists)
        for nn in range(len(self.gnn_layers)):
            gnn = self.gnn_layers[nn]
            node_feats_with_times = []
            original_size = []
            
            for t in range(self.timestep):
                g, node_feats, expanded_dists = set_of_g[t]
                node_feats = gnn(g, node_feats, expanded_dists)
                # node_feats_with_times.append(node_feats.flatten())
                node_feats_with_times.append(node_feats)
                
                original_size.append(node_feats.size())
                set_of_g[t][1] = node_feats
                
            # print(node_feats.size())
            # node_feats_with_times = torch.cat(node_feats_with_times)
            node_feats_with_times = torch.stack(node_feats_with_times,dim=2)
            
            time_mlp = self.gnn_layers_timestep[nn]
            unbind_temp = torch.unbind(node_feats_with_times,dim=0) 
            output = []
            # print(node_feats_with_times.size())
            # print(unbind_temp[0].size())
            for node_along_time in unbind_temp:
                output.append(time_mlp(node_along_time))
            
            output = torch.stack(output,dim=0)
            ## Node feature끼리 interaction하게 만들자
            # reconstructed = []
            # start_idx = 0
            # for size in original_size:
            #     tensor_size = size.numel()
            #     tensor = output[start_idx:start_idx+tensor_size]
            #     tensor = tensor.view(size)
            #     reconstructed.append(tensor)
            #     start_idx += tensor_size
            
            reconstructed = torch.unbind(output, dim=2)
            # print(reconstructed[0].size())
                        
            for t in range(self.timestep):
                set_of_g[t][1] = reconstructed[t]
        
        node_feats_with_times = []
        for t in range(self.timestep):
            node_feats_with_times.append(set_of_g[t][1])
        node_feats_with_times = torch.stack(node_feats_with_times,dim=2)
        # print(node_feats_with_times.size())
        unbind_temp = torch.unbind(node_feats_with_times,dim=0) 
        output = []
        # print(unbind_temp[0].size())
        for node_along_time in unbind_temp:
            output.append(self.timereadout(node_along_time))
        # print(output[0].size())
        res =  torch.stack(output,dim=0).squeeze()
        # print(res.size())

        return res

# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SchNet
# pylint: disable= no-member, arguments-differ, invalid-name
import torch.nn as nn

from dgl.nn.pytorch.conv.cfconv import ShiftedSoftplus

__all__ = ['SchNetPredictor']

# pylint: disable=W0221
class SchNetPredictor(nn.Module):
    """SchNet for regression and classification on graphs.

    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.

    Parameters
    ----------
    node_feats : int
        Size for node representations to learn. Default to 64.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of hidden representations for the i-th interaction
        (gnn) layer. ``len(hidden_feats)`` equals the number of interaction (gnn) layers.
        Default to ``[64, 64, 64]``.
    classifier_hidden_feats : int
        (Deprecated, see ``predictor_hidden_feats``) Size for hidden representations in the
        classifier. Default to 64.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    num_node_types : int
        Number of node types to embed. Default to 100.
    cutoff : float
        Largest center in RBF expansion. Default to 30.
    gap : float
        Difference between two adjacent centers in RBF expansion. Default to 0.1.
    predictor_hidden_feats : int
        Size for hidden representations in the output MLP predictor. Default to 64.
    """
    def __init__(self, node_feats=64, hidden_feats=None, classifier_hidden_feats=64, n_tasks=1,
                 num_node_types=100, cutoff=30., gap=0.1, predictor_hidden_feats=64):
        super(SchNetPredictor, self).__init__()

        if predictor_hidden_feats == 64 and classifier_hidden_feats != 64:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        self.gnn = SchNetGNN_timestep(node_feats, hidden_feats, num_node_types, cutoff, gap)
        self.readout = MLPNodeReadout(node_feats, predictor_hidden_feats, n_tasks,
                                      activation=ShiftedSoftplus())

    def forward(self, g, node_types, edge_dists):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_types : int64 tensor of shape (V)
            Node types to embed, V for the number of nodes.
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_types, edge_dists)
        return self.readout(g, node_feats)