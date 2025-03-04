import os
from pathlib import Path
import sys
parent_path = Path(os.getcwd())

#import torch
#import torch as th
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra
import numpy as np
#import torch_geometric.datasets as GeoData
#import numba
#from sklearn.osition import PCA
import math
import scipy.sparse


def tree_bits(depth):
    # children of i are 2i+1 and 2i+2, parent is (i-1)//2
    X = np.zeros((2**depth-1, 2**depth-2))
    for i in range(2**(depth-1)-1):
        X[2*i+1] = X[i]
        X[2*i+2] = X[i]
        X[2*i+1, 2*i] = 1
        X[2*i+1, 2*i+1] = 0
        X[2*i+2, 2*i] = 0
        X[2*i+2, 2*i+1] = 1
    return X

def tree_block_bits(depth, blocksize, extra_dim=4):
    X_tree = tree_bits(depth)
    X = np.stack([X_tree for i in range(blocksize)]).transpose(1,0,2).reshape((2**depth-1)*blocksize, 2**depth-2)
    X = np.concatenate((X, np.random.rand((2**depth-1)*blocksize, extra_dim)), axis=1)
    return X

class Fuchsian_Hyp2: # H^2 represented by uniform tiling based on Fuchsian group
    def __init__(self, num_points, raw_data, n_neighbors=15, metric="manhattan"):
        self.num_points = num_points
        #self.x = np.zeros((num_points, 2))
        self.x = np.random.randn(num_points, 2)*0.2
        self.hx = self.x2hx()
        self.g = np.tile(np.eye(3), (num_points,1,1)).astype(np.int32)
        self.xgrad = np.zeros((num_points, 2))
        self.xgrad_contra = np.zeros((num_points, 2))
        self.x_single = None

        self.n_neighbors = n_neighbors
        self.raw_data = raw_data
        #self.UMAP = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, metric=metric, output_metric="euclidean")
        self.fss = umap.fuzzy_simplicial_set(raw_data, n_neighbors=n_neighbors, random_state=0, metric=metric)[0]
        #self.a = self.UMAP.a
        #self.b = self.UMAP.b
        self.a = 3
        self.b = 1
        # const
        self.ga = np.array([[2., 1., 0.], [0., 0., -1.],[3., 2., 0.]])
        self.gb = np.array([[2., -1., 0.], [0., 0., -1.], [-3., 2., 0.]])
        self.ga_inv = np.array([[ 2.,  0., -1.], [-3., -0.,  2.], [-0., -1., -0.]])
        self.gb_inv = np.array([[ 2.,  0., 1.], [ 3., -0., 2.], [-0., -1., -0.]])
        self.L = np.array([[math.sqrt(3), 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.L_inv = np.array([[1./math.sqrt(3), 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.LgL = np.array([[-3., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        print("finished initialization")

    def zero_grad(self):
        self.xgrad = np.zeros((self.num_points, 2))
        self.xgrad_contra = np.zeros((self.num_points, 2))

    def x2hx(self):
        h0 = np.sqrt(1 + np.sum(self.x**2, axis=1, keepdims=True))
        hx = np.concatenate((h0, self.x), axis=1)
        return hx
    
    def normalize_tiling(self):
        m1 = np.array([2,-1]).T
        m2 = np.array([-1,2]).T
        iteration = 0
        #x_list = [self.x]
        #print("iteration: ", iteration, "self.x: ", self.x)
        self.hx = self.x2hx()
        while True:
            xsq = self.x**2
            disc = np.maximum(xsq@m1, xsq@m2)
            outoftile = np.where(disc >= 1) # typle of indices
            if len(outoftile[0]) == 0:
                break
            x1 = self.x[:, 0]
            x2 = self.x[:, 1]
            ind0 = np.where((x1 < -np.abs(x2)) * (disc >= 1))
            ind1 = np.where((x1 > np.abs(x2)) * (disc >= 1))
            ind2 = np.where((x2 < -np.abs(x1)) * (disc >= 1))
            ind3 = np.where((x2 > np.abs(x1)) * (disc >= 1))

            #hx = self.hx.copy()
            #g = self.g.copy()

            self.hx[ind0] = (self.L @ self.ga @ self.L_inv @ self.hx[ind0].T).T
            self.hx[ind1] = (self.L @ self.gb @ self.L_inv @ self.hx[ind1].T).T
            self.hx[ind2] = (self.L @ self.gb_inv @ self.L_inv @ self.hx[ind2].T).T
            self.hx[ind3] = (self.L @ self.ga_inv @ self.L_inv @ self.hx[ind3].T).T
            #self.hx = hx.copy()
            self.g[ind0] = self.g[ind0] @ self.ga_inv
            self.g[ind1] = self.g[ind1] @ self.gb_inv
            self.g[ind2] = self.g[ind2] @ self.gb
            self.g[ind3] = self.g[ind3] @ self.ga
            #self.g = g.copy()
            
            self.x = self.hx[:,1:].copy()
            iteration += 1
            if iteration % 5 == 0:
                self.hx = self.x2hx()
            if iteration > 20:
                break
            #x_list.append(self.x)
            #print("iteration: ", iteration, "self.x: ", self.x)
        self.hx = self.x2hx()
        #return x_list

    def single_tile(self):
        self.x_single = self.x.copy()
        self.hx = self.x2hx()
        for i in range(self.num_points):
            self.x_single[i] = (self.L @ self.g[i] @ self.L_inv @ self.hx[i])[1:]
        return self.x_single

    def distance(self, i, j):
        Q = - self.g[i].T @ self.LgL @ self.g[j]
        Q11 = Q[0,0]
        if abs(Q11) > 0.5:
            Qhat = Q / Q11
            A = self.L_inv @ Qhat @ self.L_inv @ self.hx[j]
            #print(self.hx[i].shape)
            #print(A.shape)
            dc = self.hx[i].T @ A
            #assert Q11 != 0
            #assert dc**2 - 1/Q11**2 > 0

            dist = np.log(Q11) + np.log(dc + np.sqrt(dc**2 - 1/Q11**2+1e-6))
            nablai = np.concatenate((self.x[i].reshape(-1,1) / self.hx[i,0], np.eye(2)), axis=1) @ A / np.sqrt(dc**2 - 1/Q11**2 + 1e-5)
            # gradi = (np.eye(2) + self.x[i]@self.x[i.T]) @ nablai # 更新と同時にしよう
            nablaj = np.concatenate((self.x[j].reshape(-1,1) / self.hx[j,0], np.eye(2)), axis=1) @ (self.L_inv @ Qhat.T @ self.L_inv) @ self.hx[i] / np.sqrt(dc**2 - 1/Q11**2 + 1e-5)
            return dist, nablai, nablaj
        else:
            LQL = self.L_inv @ Q @ self.L_inv
            d = self.hx[i] @ LQL @ self.hx[j]
            dist = np.arccosh(d)
            nablai = np.concatenate((self.x[i].reshape(-1,1) / self.hx[i,0], np.eye(2)), axis=1) @ LQL @ self.hx[j] / np.sqrt(d**2 - 1 + 1e-5)
            nablaj = np.concatenate((self.x[j].reshape(-1,1) / self.hx[j,0], np.eye(2)), axis=1) @ LQL.T @ self.hx[i] / np.sqrt(d**2 - 1 + 1e-5)
            return dist, nablai, nablaj    
    
    def initialize_UMAP(self, dev=1):
        flat_embed = umap.UMAP(n_neighbors=self.n_neighbors, min_dist=0.1, n_components=2, metric="manhattan", output_metric="euclidean").fit_transform(self.raw_data)
        flat_embed = flat_embed - np.mean(flat_embed, axis=0)
        flat_embed = flat_embed / np.sqrt(np.mean(flat_embed**2))
        self.x = flat_embed * dev
        self.hx = self.x2hx()
        self.normalize_tiling()

    def UMAPloss(self):
        n_edges = self.fss.shape[1]
        n_all_fedge = self.num_points * (self.num_points - 1) // 2
        negative_samples_coef = 5
        n_negative_samples = n_edges * negative_samples_coef
        #nu = scipy.sparse.csr_matrix((self.num_points, self.num_points))

        a = self.a
        b = self.b

        KLloss = 0
        # loss for each edge
        rows, cols = self.fss.nonzero()
        for r, c, mu in zip(rows, cols, self.fss.data):
            dist, nablar, nablac = self.distance(r, c)
            D2b = dist**(2*b)
            nu = 1 / (1 + self.a * D2b)
            KLloss += mu * np.log(mu) - mu * np.log(nu)
            self.xgrad[r] += mu *2*a*b*D2b/(dist+1e-9) / (1 + a*D2b) * nablar
            self.xgrad[c] += mu *2*a*b*D2b/(dist+1e-9) / (1 + a*D2b) * nablac

        # loss for each non-edge
        coef = n_all_fedge / n_negative_samples
        for i in range(n_negative_samples):
            r = np.random.randint(self.num_points)
            c = np.random.randint(self.num_points)
            mu = self.fss[r,c]
            if mu > 0.99:
                continue
            dist, nablar, nablac = self.distance(r, c)
            D2b = dist**(2*b)  
            nu = 1 / (1 + self.a * D2b)
            if nu > 0.99:
                continue
            KLloss += coef * (1-mu) * np.log(1-mu) - coef * (1-mu) * np.log(1-nu)
             
            self.xgrad[r] -= coef * (1-mu) * 2*b / (1 + a*D2b) / (dist + 1e-6) * nablar
            self.xgrad[c] -= coef * (1-mu) * 2*b / (1 + a*D2b) / (dist + 1e-6) * nablac
        return KLloss
    
    def update_contravariant_grad(self, coef=1):
        for i in range(self.num_points):
            self.xgrad_contra[i] = self.xgrad[i] @ (np.eye(2) + self.x[i].reshape(-1,1) @ self.x[i].reshape(1,-1)) * coef
    
    def fit(self, n_iter, lr = 0.1, initialize="None", dev=1, optimizer="SGD", alpha=0.99, eps=1e-8, momentum=0.9):
        loss_list = []
        if initialize == "UMAP":
            self.initialize_UMAP(dev)
        elif initialize == "spectral":
            pca = PCA(n_components=2)
            embed = pca.fit_transform(self.raw_data)
            embed = embed - np.mean(embed, axis=0)
            embed = embed / np.sqrt(np.mean(embed**2)) * dev
            self.x = embed
            self.hx = self.x2hx()
            self.normalize_tiling()            
        else:
            self.hx = self.x2hx()
            self.normalize_tiling()
    
        if optimizer == "RMSProp":
            alpha = 0.99
            eps = 1e-8

        self.normalize_tiling()
        for i in range(n_iter):
            self.zero_grad()
            loss = self.UMAPloss()
            loss_list.append(loss)
            self.update_contravariant_grad(coef=1/self.num_points)
            #self.x -= (1 - i / n_iter) * self.xgrad_contra
            if optimizer == "SGD":
                self.x -= lr * self.xgrad_contra
            elif optimizer == "RMSProp":
                pass
            self.x -= lr * self.xgrad_contra
            self.normalize_tiling()
            if i % 20 == 0 or i == n_iter - 1:
                xgrad_contra_norm = np.mean(np.linalg.norm(self.xgrad_contra, axis=1))
                print("iteration: ", i, "loss: ", loss, "grad norm: ", xgrad_contra_norm)

        return loss_list

            
