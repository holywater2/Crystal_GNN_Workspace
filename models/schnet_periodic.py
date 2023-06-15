# from dgllife.model import SchNetPredictor
from dgllife.model.gnn.schnet import RBFExpansion, Interaction

import numpy as np
import torch
import torch.nn as nn
import dgl.function as fn

# from dgl.nn.pytorch import CFConv
# from dgl.nn.pytorch.conv.cfconv import ShiftedSoftplus
from dgllife.model.readout import MLPNodeReadout

import torch.nn.init as init

class ShiftedSoftplus(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{SSP}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) - \log(\text{shift})

    Attributes
    ----------
    beta : int
        :math:`\beta` value for the mathematical formulation. Default to 1.
    shift : int
        :math:`\text{shift}` value for the mathematical formulation. Default to 2.
    """

    def __init__(self, beta=1, shift=2, threshold=20):
        super(ShiftedSoftplus, self).__init__()

        self.shift = shift
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, inputs):
        """

        Description
        -----------
        Applies the activation function.

        Parameters
        ----------
        inputs : float32 tensor of shape (N, *)
            * denotes any number of additional dimensions.

        Returns
        -------
        float32 tensor of shape (N, *)
            Result of applying the activation function to the input.
        """
        return self.softplus(inputs) - np.log(float(self.shift))


class CFConvPBC(nn.Module):
    r"""CFConv from `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__

    It combines node and edge features in message passing and updates node representations.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} h_j^{l} \circ W^{(l)}e_ij

    where :math:`\circ` represents element-wise multiplication and for :math:`\text{SPP}` :

    .. math::
        \text{SSP}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) - \log(\text{shift})

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features :math:`h_j^{(l)}`.
    edge_in_feats : int
        Size for the input edge features :math:`e_ij`.
    hidden_feats : int
        Size for the hidden representations.
    out_feats : int
        Size for the output representations :math:`h_j^{(l+1)}`.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import CFConv
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> nfeat = th.ones(6, 10)
    >>> efeat = th.ones(6, 5)
    >>> conv = CFConv(10, 5, 3, 2)
    >>> res = conv(g, nfeat, efeat)
    >>> res
    tensor([[-0.1209, -0.2289],
            [-0.1209, -0.2289],
            [-0.1209, -0.2289],
            [-0.1135, -0.2338],
            [-0.1209, -0.2289],
            [-0.1283, -0.2240]], grad_fn=<SubBackward0>)
    """

    def __init__(self, node_in_feats, edge_in_feats, hidden_feats, out_feats):
        super(CFConvPBC, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Linear(edge_in_feats, hidden_feats),
            ShiftedSoftplus(),
            nn.Linear(hidden_feats, hidden_feats),
            ShiftedSoftplus(),
        )
        self.project_node = nn.Linear(node_in_feats, hidden_feats)
        self.project_out = nn.Sequential(
            nn.Linear(hidden_feats, out_feats), ShiftedSoftplus()
        )

    def forward(self, g, node_feats, edge_feats):
        """

        Description
        -----------
        Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        node_feats : torch.Tensor or pair of torch.Tensor
            The input node features. If a torch.Tensor is given, it represents the input
            node feature of shape :math:`(N, D_{in})` where :math:`D_{in}` is size of
            input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph,
            the pair must contain two tensors of shape :math:`(N_{src}, D_{in_{src}})` and
            :math:`(N_{dst}, D_{in_{dst}})` separately for the source and destination nodes.

        edge_feats : torch.Tensor
            The input edge feature of shape :math:`(E, edge_in_feats)`
            where :math:`E` is the number of edges.

        Returns
        -------
        torch.Tensor
            The output node feature of shape :math:`(N_{out}, out_feats)`
            where :math:`N_{out}` is the number of destination nodes.
        """
        with g.local_scope():
            if isinstance(node_feats, tuple):
                node_feats_src, _ = node_feats
            else:
                node_feats_src = node_feats
            g.srcdata["hv"] = self.project_node(node_feats_src)
            g.edata["he"] = self.project_edge(edge_feats)
            g.update_all(fn.u_mul_e("hv", "he", "m"), fn.sum("m", "h"))
            return self.project_out(g.dstdata["h"])


class RBFExpansion(nn.Module):
    r"""Expand distances between nodes by radial basis functions.

    .. math::
        \exp(- \gamma * ||d - \mu||^2)

    where :math:`d` is the distance between two nodes and :math:`\mu` helps centralizes
    the distances. We use multiple centers evenly distributed in the range of
    :math:`[\text{low}, \text{high}]` with the difference between two adjacent centers
    being :math:`gap`.

    The number of centers is decided by :math:`(\text{high} - \text{low}) / \text{gap}`.
    Choosing fewer centers corresponds to reducing the resolution of the filter.

    Parameters
    ----------
    low : float
        Smallest center. Default to 0.
    high : float
        Largest center. Default to 30.
    gap : float
        Difference between two adjacent centers. :math:`\gamma` will be computed as the
        reciprocal of gap. Default to 0.1.
    """
    def __init__(self, low=0., high=30., gap=0.1):
        super(RBFExpansion, self).__init__()

        num_centers = int(np.ceil((high - low) / gap))
        self.centers = np.linspace(low, high, num_centers)
        self.centers = nn.Parameter(torch.tensor(self.centers).float(), requires_grad=False)
        self.gamma = 1 / gap

    def reset_parameters(self):
        """Reinitialize model parameters."""
        device = self.centers.device
        self.centers = nn.Parameter(
            self.centers.clone().detach().float(), requires_grad=False).to(device)

    def forward(self, edge_dists):
        """Expand distances.

        Parameters
        ----------
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (E, len(self.centers))
            Expanded distances.
        """
        radial = edge_dists - self.centers
        coef = - self.gamma
        return torch.exp(coef * (radial ** 2))

class Interaction(nn.Module):
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
    def __init__(self, node_feats, edge_in_feats, hidden_feats,max_neighbors):
        super(Interaction, self).__init__()
        self.max_neighbors = max_neighbors
        self.conv = CFConvPBC(node_feats, edge_in_feats, hidden_feats, node_feats)
        self.project_out = nn.Linear(node_feats, node_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.conv.project_edge:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.conv.project_node.reset_parameters()
        self.conv.project_out[0].reset_parameters()
        self.project_out.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
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
        node_feats = self.conv(g, node_feats, edge_feats) / self.max_neighbors
        return self.project_out(node_feats)

class SchNetPeriodicGNN(nn.Module):
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
    def __init__(self, node_feats=64, hidden_feats=None, num_node_types=100, cutoff=30., gap=0.1, max_neighbor=12):
        super(SchNetPeriodicGNN, self).__init__()

        if hidden_feats is None:
            hidden_feats = [64, 64, 64]

        self.embed = nn.Embedding(num_node_types, node_feats)
        self.rbf = RBFExpansion(high=cutoff, gap=gap)
        self.max_neighbor = max_neighbor

        n_layers = len(hidden_feats)
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                Interaction(node_feats, len(self.rbf.centers), hidden_feats[i], max_neighbors=12)
                )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.embed.reset_parameters()
        self.rbf.reset_parameters()
        for layer in self.gnn_layers:
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
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats, expanded_dists) / self.max_neighbor
        return node_feats


# pylint: disable=W0221
class SchNetPeriodicPredictor(nn.Module):
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
                 num_node_types=100, cutoff=30., gap=0.1, predictor_hidden_feats=64, max_neighbor=12):
        super(SchNetPeriodicPredictor, self).__init__()

        if predictor_hidden_feats == 64 and classifier_hidden_feats != 64:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        self.gnn = SchNetPeriodicGNN(node_feats, hidden_feats, num_node_types, cutoff, gap, max_neighbor=12)
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
