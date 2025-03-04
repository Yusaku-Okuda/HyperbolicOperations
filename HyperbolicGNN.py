# Imports
import torch
import torch as th
from torch import cosh, sinh,tanh, arctanh, arccosh, arcsinh
from torch.linalg import solve
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv, GATConv
from einops import rearrange #pip install einops
from typing import List
import random
import math
from torch.utils.data import DataLoader
import numpy as np
import itertools
from collections import defaultdict
# import os

from HyperbolicOperations import *
from modules import *
from scipy.special import beta

# using float64 is recommended

dtype = th.float32

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, act=nn.ReLU()):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        adj = th.eye(adj.size(0)).to(dtype) + adj # こっちが先なのが重要
        adj = adj.to_dense()
        D = 1/ th.sqrt(th.sum(adj, axis=1))
        adj = adj*D.view(-1, 1)*D.view(1, -1)
        #adj = th.eye(adj.size(0)).to(dtype) + adj
        adj = adj.to_sparse()
        support = torch.mm(input, self.weight)
        
        output = torch.sparse.mm(adj, support) # sparse matrix multiplication
        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN_handmade(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, actout=nn.Identity(), nlayers=2):
        super(GCN_handmade, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(GraphConvolution(in_dim, out_dim))
        else:
            self.layers.append(GraphConvolution(in_dim, hidden_dim))
            for i in range(nlayers-2):
                self.layers.append(GraphConvolution(hidden_dim, hidden_dim))
            self.layers.append(GraphConvolution(hidden_dim, out_dim, act=actout))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        for i in range(self.nlayers):
            if i != 0:
                x = self.dropout(x)
            x = self.layers[i](x, adj)
        return x

class GCN_PP(nn.Module):
    def __init__(self, in_dim, hidden_dim, F_dim, class_dim, S_dim, nlayers=2):
        super(GCN_PP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.class_dim = class_dim
        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(GraphConvolution(in_dim, F_dim))
        else:
            self.layers.append(GraphConvolution(in_dim, hidden_dim, act=nn.LeakyReLU()))
            for i in range(nlayers-2):
                self.layers.append(GraphConvolution(hidden_dim, hidden_dim, act=nn.LeakyReLU()))
            #self.layers.append(GraphConvolution(hidden_dim, F_dim, act=nn.Identity()))
            self.layers.append(GraphConvolution(hidden_dim, F_dim, act=nn.Identity()))
            self.batchnorm = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(0.5)
        #self.pre_class = MLP(S_dim, hidden_dim, S_dim, n_layers=1, act_out=nn.Tanh())
        self.classifier = MLP(F_dim+S_dim, 10, class_dim, n_layers=1, act_out=nn.Softmax(dim=-1))
        nn.init.xavier_normal_(self.classifier.layers[0].weight, gain=2)
        #self.classifier1 = DistLayer(S_dim, 10, act=nn.Identity())
        #self.classifier2 = MLP(F_dim+10,10, class_dim, n_layers=1, act_out=nn.Softmax(dim=-1))
        self.mu = None
        self.vars = None
        self.gamma = None
        self.ncluster = self.class_dim
        self.cos1 = Parameter(th.tensor([0.5]))
        self.cos2 = Parameter(th.tensor([0.5]))
        
    def forward(self, x, adj, S):
        for i in range(self.nlayers):
            x = self.layers[i](x, adj)
            if i == 0:
                #x = self.batchnorm(x)
                x = self.dropout(x)

        #S = self.pre_class(S)
        y = self.classifier(th.cat((x, S), dim=1))
        #S = self.classifier1(S)
        #y = self.classifier2(th.cat((x, S), dim=1))
        return x, y
    
    def update_gmm_covariance(self, emd, iteration=10): # use EM algorithm
        # emd: (nnodes, dim)
        with th.no_grad():
            ncluster = self.ncluster
            nnodes = emd.size(0)
            dim = emd.size(1)
            mu = th.zeros(ncluster, dim)
            vars = th.zeros(ncluster, dim, dim)
            if self.gamma is None:
                mu, cluster = KmeansPP(emd, ncluster, iteration=5)
                gamma = th.nn.functional.one_hot(cluster, ncluster).to(dtype) + th.rand(nnodes, ncluster) * 0.1
                gamma = gamma / th.sum(gamma, dim=1, keepdim=True)
                pi = th.sum(gamma, dim=0) / nnodes
            else:
                gamma = self.gamma.clone().requires_grad_(False)
                pi = th.sum(gamma, dim=0) / nnodes
            for i in range(iteration):
                sum_gamma = th.sum(gamma, dim=0) # ncluster
                print(i, sum_gamma)
                emd_ = emd.unsqueeze(1).expand(nnodes, ncluster, dim).permute(0,2,1) # nnodes, dim, ncluster
                mu = (th.sum(emd_ * gamma.unsqueeze(1).expand_as(emd_) , dim=0) / sum_gamma).transpose(0,1) # ncluster, dim
                diff = emd.unsqueeze(1).expand(nnodes, ncluster, dim) - mu # nnodes, ncluster, dim
                emd_ = diff.reshape(nnodes*ncluster, dim, 1)
                emd_ = th.bmm(emd_, emd_.permute(0,2,1)).view(nnodes, ncluster, dim, dim)
                vars = th.sum(emd_ * gamma.unsqueeze(2).unsqueeze(3).expand_as(emd_), dim=0) / (sum_gamma.unsqueeze(1).unsqueeze(2).expand(ncluster,dim,dim)-1) # ncluster, dim, dim
                # 不偏分散的な計算

                # update gamma, pi
                det = th.det(vars) # ncluster
                print("det = ", end="")
                print(det)
                inv_vars_diff = solve(vars, diff.permute(1,2,0)).permute(2,0,1) # nnodes, ncluster, dim
                prob = th.exp(-0.5 * th.sum(diff * inv_vars_diff, dim=-1)) / th.sqrt((2*math.pi)**dim * det) * pi # nnodes, ncluster
                gamma = prob / th.sum(prob, dim=1, keepdim=True)
                pi = th.sum(gamma, dim=0) / nnodes
            self.det = det
            self.mu = mu
            self.vars = vars
            self.gamma = gamma
            self.ncluster = ncluster
        
    def update_gmm_scalar(self, emd, iteration=10): # use EM algorithm, vars is scalar
        # emd: (nnodes, dim)
        with th.no_grad():
            ncluster = self.ncluster
            nnodes = emd.size(0)
            dim = emd.size(1)
            mu = th.zeros(ncluster, dim) if self.mu is None else self.mu
            vars = th.zeros(ncluster) if self.vars is None else self.vars
            if self.mu is None:
                mu, cluster = KmeansPP(emd, ncluster, iteration=5)
                gamma = th.nn.functional.one_hot(cluster, ncluster).to(dtype)
                gamma = gamma / th.sum(gamma, dim=1, keepdim=True)
                pi = th.sum(gamma, dim=0) / nnodes
            else:
                gamma = self.gamma.clone().requires_grad_(False)
                pi = th.sum(gamma, dim=0) / nnodes

            for i in range(iteration):
                sum_gamma = th.sum(gamma, dim=0) # ncluster

                emd_ = emd.unsqueeze(1).expand(nnodes, ncluster, dim).permute(0,2,1) # nnodes, dim, ncluster
                mu = (th.sum(emd_ * gamma.unsqueeze(1).expand_as(emd_) , dim=0) / sum_gamma).transpose(0,1) # ncluster, dim
                diff = emd.unsqueeze(1).expand(nnodes, ncluster, dim) - mu # nnodes, ncluster, dim
                vars = th.sum(diff**2 * gamma.unsqueeze(2).expand_as(diff), dim=(0, 2)) / sum_gamma # ncluster
                vars = th.clamp(vars, min=0.1)

                # update gamma, pi
                prob = th.exp(-0.5 * th.sum(diff**2, dim=-1) / vars) / th.sqrt(2*math.pi*vars)**dim * pi # nnodes, ncluster
                gamma = prob / th.sum(prob, dim=1, keepdim=True)
                pi = th.sum(gamma, dim=0) / nnodes

            print("{} var = ".format(i), end="")
            print(vars)
            print("sum gamma = ", end="")
            print(sum_gamma)
        self.mu = mu
        self.vars = vars
        self.gamma = gamma
        self.ncluster = ncluster
                
    def update_gmm_ss_cov(self, emd, y):
        # emd: (nnodes, dim), y: (nnodes, ncluster)
        with th.no_grad():
            ncluster = self.ncluster
            nnodes = emd.size(0)
            dim = emd.size(1)
            emd_ = emd.unsqueeze(1).expand(nnodes, ncluster, dim).permute(0,2,1) # nnodes, dim, ncluster
            mu = (th.sum(emd_ * y.unsqueeze(1).expand_as(emd_) , dim=0) / th.sum(y, dim=0)).transpose(0,1) # ncluster, dim
            diff = emd_.permute(0,2,1) - mu # nnodes, ncluster, dim
            emd_ = diff.reshape(nnodes*ncluster, dim, 1)
            emd_ = th.bmm(emd_, emd_.permute(0,2,1)).view(nnodes, ncluster, dim, dim)
            vars = th.sum(emd_ * y.unsqueeze(2).unsqueeze(3).expand_as(emd_), dim=0) / th.sum(y, dim=0).unsqueeze(1).unsqueeze(2).expand(ncluster,dim,dim) # ncluster, dim, dim
            det = th.det(vars) # ncluster
            print("det = ", end="")
            print(th.pow(det, 1/dim))
            self.det = det
            self.mu = mu
            self.vars = vars
            self.ncluster = ncluster
            self.gamma = y

    def loglikelihood_ss(self, emd, y):
        self.gamma = y
        if self.vars.dim() == 1:
            return self.log_likelihood_scalar(emd)
        nnodes = emd.size(0)
        dim = emd.size(1)
        ncluster = self.ncluster
        diff = emd.unsqueeze(1).expand(nnodes, ncluster, dim) - self.mu # nnodes, ncluster, dim
        inv_vars_diff = solve(self.vars, diff.permute(1,2,0)).permute(2,0,1) # nnodes, ncluster, dim
        prob = th.exp(-0.5 * th.sum(diff * inv_vars_diff, dim=-1)) / th.sqrt((2*math.pi)**dim * self.det) * th.sum(self.gamma, dim=0)/nnodes # nnodes, ncluster
        loglik = th.mean(th.log(th.sum(prob, dim=1)))
        return loglik

    def loglikelihood(self, emd):
        if self.vars.dim() == 1:
            return self.loglikelihood_scalar(emd)
        nnodes = emd.size(0)
        dim = emd.size(1)
        ncluster = self.ncluster
        diff = emd.unsqueeze(1).expand(nnodes, ncluster, dim) - self.mu # nnodes, ncluster, dim
        inv_vars_diff = solve(self.vars, diff.permute(1,2,0)).permute(2,0,1) # nnodes, ncluster, dim
        prob = th.exp(-0.5 * th.sum(diff * inv_vars_diff, dim=-1)) / th.sqrt((2*math.pi)**dim * self.det) * th.sum(self.gamma, dim=0)/nnodes # nnodes, ncluster
        loglik = th.mean(th.log(th.sum(prob, dim=1)))
        return loglik
    
    def loglikelihood_scalar(self, emd):
        nnodes = emd.size(0)
        dim = emd.size(1)
        ncluster = self.ncluster
        diff = emd.unsqueeze(1).expand(nnodes, ncluster, dim) - self.mu
        prob = th.exp(-0.5 * th.sum(diff**2, dim=-1) / self.vars) / th.sqrt(2*math.pi*self.vars)**dim * th.sum(self.gamma, dim=0)/nnodes
        loglik = th.mean(th.log(th.sum(prob, dim=1)))
        return loglik

    def edge_reconst_loss(self, emd, edge_index, edge2hop):
        normed_emd = emd / th.norm(emd, dim=1, keepdim=True)
        loss_1 = th.mean(nn.functional.softplus(- th.sum(normed_emd[edge_index[0]]* normed_emd[edge_index[1]], dim=-1))) # 1-hop loss
        loss_2 = th.mean(nn.functional.softplus(- th.sum(normed_emd[edge2hop[0]]* normed_emd[edge2hop[1]], dim=-1))) # 2-hop loss
        return loss_1 + loss_2

class GCNmass(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, actout=nn.Identity(), nlayers=2, no_mass=False):
        super(GCNmass, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(GCNConv(in_dim, out_dim, cached=True))
        else:
            self.layers.append(GCNConv(in_dim, hidden_dim, cached=True))
            for i in range(nlayers-2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=True))
            self.layers.append(GCNConv(hidden_dim, out_dim + 1, cached=True))
        self.dropout = nn.Dropout(0.5)
        self.mass_coeff = Parameter(th.tensor([5.0]))
        self.no_mass = no_mass

    def forward(self, x, edge_index):
        for i in range(self.nlayers):
            x = self.layers[i](x, edge_index)
            if i != self.nlayers - 1:
                x = F.relu(x)
                x = self.dropout(x)
            pos = x[:, :self.out_dim]
            if not self.no_mass:
                mass = th.exp(x[:, self.out_dim:] ) # * self.mass_coeff)
            else:
                mass = th.ones(x.size(0), 1).to(dtype)
        return pos, mass

def Kmeans(emd, ncluster, iteration=10):
    nnodes = emd.size(0)
    dim = emd.size(1)
    rand_idx = th.randperm(nnodes)[:ncluster]
    mu = emd[rand_idx]
    for _ in range(iteration):
        diff = emd.unsqueeze(1).expand(nnodes, ncluster, dim) - mu # nnodes, ncluster, dim
        dist = th.sum(diff**2, dim=-1)
        cluster = dist.argmin(dim=-1)
        for i in range(ncluster):
            if th.sum(cluster==i) == 0:
                print(_)
                return mu, cluster
            mu[i] = th.mean(emd[cluster==i], dim=0)
    return mu, cluster

def KmeansPP(emd, ncluster, iteration=100):
    nnodes = emd.size(0)
    dim = emd.size(1)
    rand_idx = th.randint(nnodes, (1,))
    mu = emd[rand_idx]
    for i in range(ncluster-1):
        diff = emd.unsqueeze(1).expand(nnodes, mu.shape[0], dim) - mu # nnodes, ncluster, dim
        dist = th.sum(diff**2, dim=-1)
        prob, idx = th.min(dist, dim=-1)
        prob = prob / th.sum(prob)
        rand_idx = th.multinomial(prob, 1)
        mu = th.cat((mu, emd[rand_idx]), dim=0)
    for _ in range(iteration):
        diff = emd.unsqueeze(1).expand(nnodes, ncluster, dim) - mu # nnodes, ncluster, dim
        dist = th.sum(diff**2, dim=-1)
        cluster = dist.argmin(dim=-1)
        for i in range(ncluster):
            if th.sum(cluster==i) == 0:
                print(_)
                return mu, cluster
            mu[i] = th.mean(emd[cluster==i], dim=0)
    return mu, cluster
    
class GNN_Aug(nn.Module):
    def __init__(self, in_dim, hidden_dim, emd_dim, out_dim, nlayers=2):
        super(GNN_Aug, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.GCN = GCN_handmade(in_dim, hidden_dim, emd_dim, nlayers)
        self.classifier = MLP(emd_dim, hidden_dim, out_dim, 2, act_out=nn.Softmax(dim=-1))
        self.softmax = nn.Softmax(dim=-1)
        self.vars = None
        self.gamma = None
        self.ncluster = self.out_dim

class DistLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.Identity()):
        super(DistLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(th.randn(out_dim, in_dim) * 3)
        self.bias = Parameter(th.ones(out_dim))
        self.act = act
        self.bias.register_hook(lambda grad: grad * 1)
        self.weight.register_hook(lambda grad: grad * 1)        

    def forward(self, x):
        #dist = th.exp(-th.sum((x.unsqueeze(1) - self.weight)**2, dim=-1) / self.bias/2) / th.sqrt(2*math.pi*self.bias)**self.in_dim
        dist = -th.sqrt(th.sum((x.unsqueeze(1) - self.weight)**2, dim=-1)) + self.bias
        return self.act(dist)

class Embedding(nn.Module):
    def __init__(self, n_nodes, edge, dim=10):
        adj_list = defaultdict(list)
        for i, j in zip(edge[0], edge[1]):
            adj_list[i.item()].append(j.item())
            adj_list[j.item()].append(i.item())
        adjacency_list = [adj_list[i] for i in range(n_nodes)]
        #self.adjacency_list = adjacency_list
        x = th.randn(n_nodes, dim) * 0.1

        bsz = 1000
        iteration = 1000
        for i in range(iteration):
            batch_edge = th.randint(0, n_nodes, (2, bsz))


class VAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, z_dim, nlayers=2):
        super(VAE, self).__init__()
        self.encoder = MLP(in_dim, hidden_dim, z_dim, nlayers, act_out=nn.Identity())
        self.decoder = MLP(z_dim, hidden_dim, in_dim, nlayers, act_out=nn.Identity())
        self.mu = th.zeros(in_dim)
        self.logstd = th.zeros(1)

    def forward(self, x):
        z = self.encoder((x-self.mu)/self.std)
        x_hat = self.decoder(z) * th.exp(self.logstd) + self.mu
        return x_hat, z
    
    def sample(self, z):
        return self.decoder(z) * th.exp(self.logstd) + self.mu


class GraphConvHyp(Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU(), to_tan=False):
        super(GraphConvHyp, self).__init__()
        self.in_features = in_dim
        self.out_features = out_dim
        self.act = act
        self.FC = PoincareFC(in_dim=in_dim, out_dim=out_dim, act=act, to_tan=to_tan)
        
    def forward(self, x, adj):
        adj = th.eye(adj.size(0)).to(dtype) + adj
        D = 1/ th.sqrt(th.sum(adj, axis=1))
        adj = adj*D.view(-1, 1)*D.view(1, -1)
        adj = adj.to_sparse()
        aggr = self.gyrmidpoint(x, adj)
        output = self.FC(aggr)
        return output

    def gyrmidpoint(self, x, weight):
        weight = weight.to_dense()
        r2 = th.sum(x**2, dim=-1, keepdim=True)
        gamma = 2/(1-r2)
        gamma = th.clamp(gamma, max=1e7)
        num = (weight * gamma)
        den = th.sum(weight*(gamma-1), dim=-1, keepdim=True)
        coef = (num/den).to_sparse()
        x = th.sparse.mm(coef, x)
        return MMulti(x, 1/2)
    

class HyperbolicGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,actout=nn.Identity() ,nlayers=2):
        super(HyperbolicGCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(GraphConvHyp(in_dim, out_dim, act=actout))
        else:
            self.layers.append(GraphConvHyp(in_dim, hidden_dim))
            for i in range(nlayers-2):
                self.layers.append(GraphConvHyp(hidden_dim, hidden_dim))
            self.layers.append(GraphConvHyp(hidden_dim, out_dim, act=actout, to_tan=True))
        
    def forward(self, x, adj):
        for i in range(self.nlayers):
            x = self.layers[i](x, adj)
        return x
    

class EdgeSampler():
    def __init__(self, datax, sparse_edge):
        self.sparse_edge = sparse_edge
        sparse_2hop = torch.sparse.mm(sparse_edge, sparse_edge)
        self.neighbor_set = set(tuple(sparse_2hop._indices()[:, i].tolist()) for i in range(sparse_edge._nnz()))
        sparse_2hop = th.sparse_coo_tensor(sparse_2hop.indices(), th.ones(sparse_2hop.indices().shape[1]), (datax.shape[0], datax.shape[0]))
        sparse_2hop = sparse_2hop - sparse_edge
        sparse_2hop = sparse_2hop - th.sparse_coo_tensor(th.stack([th.arange(datax.shape[0]), th.arange(datax.shape[0])]), th.ones(datax.shape[0]), (datax.shape[0], datax.shape[0]))
        sparse_2hop = sparse_2hop.coalesce()
        nonzero_mask = sparse_2hop._values() > 0.001
        self.sparse_2hop = th.sparse_coo_tensor(sparse_2hop._indices()[:, nonzero_mask], sparse_2hop._values()[nonzero_mask], (datax.shape[0], datax.shape[0]))
        
    def sample_edge(self, nsamples): # shape: 2, nsamples
        n_edges = self.sparse_edge._nnz()
        edge_idx = self.sparse_edge._indices()
        sample_idx = torch.randint(0, n_edges, (nsamples,))
        sampled_edges = edge_idx[:, sample_idx]
        return sampled_edges

    def sample_non_edge(self, nsamples): # shape: 2, nsamples
        n_edges = self.sparse_edge._nnz()
        edge_idx = self.sparse_edge._indices()
        n_nodes = self.sparse_edge.shape[0]
        non_edges = []
        while len(non_edges) < nsamples: # 2, nsamples
            i, j = random.randint(0, n_nodes - 1), random.randint(0, n_nodes - 1)
            if i == j or (i, j) in self.neighbor_set or (j, i) in self.neighbor_set:
                continue
            non_edges.append([i, j])

        return torch.tensor(non_edges).t()
    
    def sample_2hop_edge(self, nsamples): # shape: 2, nsamples
        n_edges = self.sparse_2hop._nnz()
        edge_idx = self.sparse_2hop._indices()
        sample_idx = torch.randint(0, n_edges, (nsamples,))
        sampled_edges = edge_idx[:, sample_idx]
        return sampled_edges

    def all_edges(self):
        return self.sparse_edge._indices()

    def all_2hop_edges(self):
        return self.sparse_2hop._indices()

class GradientGating_Unit(nn.Module):
    def __init__(self, dim, act=nn.ReLU(), GNN="GCN"):
        super(GradientGating_Unit, self).__init__()
        self.dim = dim
        self.act = act
        if GNN == "GCN":
            self.GNN = GCNConv(dim, dim, cached=False)
        elif GNN == "GAT":
            self.GNN = GATConv(dim, dim, heads=3, concat=False, dropout=0.5)
        #self.GNN2 = GCNConv(dim, dim, cached=True)
        
    def forward(self, x, edge_index):
        y = self.GNN(x, edge_index)
        y = self.act(y)
        # calculate dirichlet energy of y
        D = torch.sparse.sum(edge_index, dim=1).to_dense()
        y2 = y**2
        #y_agg = torch.sparse.mm(edge_index, y).clone() # 疎行列にすると勾配が計算出来ない
        #y2_agg = torch.sparse.mm(edge_index, y2).clone()
        dense_edge_index = edge_index.to_dense()
        y_agg = torch.mm(dense_edge_index, y)
        y2_agg = torch.mm(dense_edge_index, y2)
        dirichlet_energy = th.tanh(y2*D.unsqueeze(1) + y2_agg - 2*y*y_agg)

        #z = self.act(self.GNN2(x, edge_index))
        
        return x * (1 - dirichlet_energy) + y * dirichlet_energy
        #return x + y * dirichlet_energy
    

class GradientGating(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, nlayers=2, GNN="GCN"):
        super(GradientGating, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, out_dim)
        for i in range(nlayers):
            self.layers.append(GradientGating_Unit(hidden_dim, GNN=GNN))
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index):
        x = self.encoder(x)
        for i in range(self.nlayers):
            #print(x.shape, edge_index.size())
            x = self.dropout(x)
            x = self.layers[i](x, edge_index)
        x = self.decoder(x)
        return x

class Arbitrary_embedding(nn.Module):
    def __init__(self, N, out_dim):
        super(Arbitrary_embedding, self).__init__()
        self.out_dim = out_dim
        self.N = N
        self.embedding = nn.Parameter(th.randn(N, out_dim) * 0.1)
        
    def forward(self, x, edge_index):
        return self.embedding
