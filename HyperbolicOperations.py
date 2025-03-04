# Imports
import torch
import torch as th
from torch import cosh, sinh,tanh, arctanh, arccosh, arcsinh
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange #pip install einops
from typing import List
import random
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV3 #pip install timm 
from tqdm import tqdm #pip install tqdm
import matplotlib.pyplot as plt #pip install matplotlib
import torch.optim as optim
import numpy as np
# import os

# using float64 is recommended
dtype = th.float32

#class Lorentz(th.Tensor):



def expP(x, v, k=1):# 指数写像 (Poincare Model)
    # x: base point, v: tangent vector
    # 曲率が-k

    if isinstance(k, float) or isinstance(k, int):
        if x.dim() == 2:
            k = (th.ones(x.shape[0])*k).view(-1,1)
    elif isinstance(k, list):
        k = th.tensor(k).view(-1, 1)
    elif k.dim() == 1:
        k = k.view(-1, 1)
    else:
        print("Error: Dimension Error: expP")
        raise ValueError

    k = k.to(dtype)
    sqrtk = th.sqrt(k)
    if v.dim() == 1:
        nv = th.norm(v, p=2)
        nx = th.norm(x, p=2)
        return MAdd(x, v / sqrtk * (tanh(sqrtk * nv / 2 / (1 - k*nx**2)) / th.maximum(nv, th.ones(1)*1e-6)) , k)
    nv = th.norm(v, dim=1, p=2).view(-1, 1)
    nx = th.norm(x, dim=1, p=2).view(-1, 1)

    return MAdd(x, v / sqrtk * (tanh((sqrtk * nv / 2 / (1-k*nx**2))) / th.maximum(nv, th.ones(nv.shape)*1e-6) ) , k)
    # /2 が多分必要なはず

def expP0(v, k=1):
    if isinstance(k, float) or isinstance(k, int):
        if v.dim() >= 2:
            k = (th.ones(v.shape[0])*k).view(-1,1)
    elif isinstance(k, list):
        k = th.tensor(k).view(-1, 1)
    elif k.dim() == 1:
        k = k.view(-1, 1)
    else:
        print("Error: Dimension Error: expP")
        raise ValueError

    k = k.to(dtype)
    sqrtk = th.sqrt(k)
    if v.dim() == 1:
        nv = th.norm(v, p=2)
        return MAdd(th.zeros_like(v), v / sqrtk * (tanh(sqrtk * nv / 2) / th.maximum(nv, th.ones(1)*1e-6)) , k)
    nv = th.norm(v, dim=1, p=2).view(-1, 1)

    #return MAdd(th.zeros_like(v), v / sqrtk * (tanh((sqrtk * nv / 2)) / th.maximum(nv, th.ones(nv.shape)*1e-6) ) , k)
    return v / sqrtk * (tanh((sqrtk * nv / 2)) / th.maximum(nv, th.ones(nv.shape)*1e-6) )
    # /2 が多分必要なはず

def logP(x, y, k=1):# 対数写像 (Poincare Model)
    # x: base point, y: target point, return: \in T_x M
    if isinstance(k, float) or isinstance(k, int):
        if x.dim() == 2:
            k = (th.ones(x.shape[0])*k).view(-1,1)
    elif isinstance(k, list):
        k = th.tensor(k).view(-1, 1)
    elif k.dim() == 1:
        k = k.view(-1, 1)
    else:
        print("Error: Dimension Error: logP")
        raise ValueError   
    
    k = k.to(dtype)
    sqrtk = th.sqrt(k)
    if y.dim() == 1:
        z = MAdd(-x, y, k)
        nz = th.norm(z, p=2)
        x2 = th.sum(x*x)
    else:
        if th.max(th.sum(y**2, dim=1)) < 1:
            print("dim = {}".format(y.shape))
        assert th.max(th.sum(x**2, dim=1)) < 1
        assert th.max(th.sum(y**2, dim=1)) < 1
        z = MAdd(-x, y, k)
        #print(z)
        nz = th.norm(z, p=2, dim=1).view(-1, 1)
        x2 = th.sum(x*x, dim=1).view(-1, 1)
    #print(z / th.maximum(nz, th.ones(nz.shape)*1e-8) * (1 - sqrtk*x2) / sqrtk * arctanh(sqrtk*nz))
    #print("logP, nz = {}".format(nz.view(-1)))
    return z / th.maximum(nz, th.ones(nz.shape, dtype=dtype)*1e-8) * (1 - sqrtk*x2) / sqrtk * arctanh(sqrtk*nz) * 2
    # *2 が必要なはず

# 双曲面モデルは第0軸の座標を省略する

def expH(x, v, k=1):# 指数写像 (双曲面モデル)
    # x \in R^n: base point, v \in R^{n+1}: tangent vector
    # 曲率が-k
    if isinstance(k, float) or isinstance(k, int):
        if x.dim() == 2:
            k = (th.ones(x.shape[0])*k).view(-1,1)
    elif isinstance(k, list):
        k = th.tensor(k).view(-1, 1)
    elif k.dim() == 1:
        k = k.view(-1, 1)
    else:
        print("Error: Dimension Error: expH")
        raise ValueError
    
    assert x.dim() == 2
    assert x.shape[1] + 1 == v.shape[1]
    sqrtk = th.sqrt(k)
    x_ = th.cat([x, th.sqrt(th.sum(x**2, dim=1) + 1/k).view(-1, 1)], dim=1)
    v_norm = th.norm(v, p=2, dim=-1).view(-1, 1) * sqrtk
    return ( th.cosh(v_norm) * x_ + th.sinh(v_norm) * v / th.maximum(v_norm, th.ones(v_norm.shape)*1e-9) )[:, :-1]

def expH0(v, k=1):# 指数写像 (双曲面モデル)
    # v \in R^n: tangent vector
    # 曲率が-k

    if isinstance(k, float) or isinstance(k, int):
        if v.dim() == 2:
            k = (th.ones(v.shape[0])*k).view(-1,1)
    elif isinstance(k, list):
        k = th.tensor(k).view(-1, 1)
    elif k.dim() == 1:
        k = k.view(-1, 1)
    else:
        print("Error: Dimension Error: expH")
        raise ValueError
    
    sqrtk = th.sqrt(k)
    v_norm = th.norm(v, p=2, dim=-1).view(-1, 1) * sqrtk
    return v / th.maximum(v_norm, th.ones(v_norm.shape)*1e-9) * th.sinh(v_norm)

    
def logH(x, y, k=1):# 対数写像 (双曲面モデル)
    # x \in R^n: base point, y \in R^n: target point, return: \in T_x M
    if isinstance(k, float) or isinstance(k, int):
        if y.dim() == 2:
            k = (th.ones(y.shape[0])*k).view(-1,1)
    elif isinstance(k, list):
        k = th.tensor(k).view(-1, 1)
    elif k.dim() == 1:
        k = k.view(-1, 1)
    else:
        print("Error: Dimension Error: expH")
        raise ValueError

    assert x.dim() == 2
    assert y.dim() == 2
    inpro = th.sum(x*y, dim=-1).view(-1, 1) * k - th.sqrt((th.sum(x**2, dim=-1).view(-1,1)*k + 1) * (th.sum(y**2, dim=-1).view(-1,1)*k + 1))
    cosh_inpro = th.arccosh(th.maximum(-inpro, th.ones(inpro.shape)))
    return cosh_inpro / sinh(cosh_inpro) * (y + x * inpro)

def logH0(y, k=1):# 対数写像 (双曲面モデル)
    # y \in R^n: target point, return: \in T_x M
    if isinstance(k, float) or isinstance(k, int):
        if y.dim() == 2:
            k = (th.ones(y.shape[0])*k).view(-1,1)
    elif isinstance(k, list):
        k = th.tensor(k).view(-1, 1)
    elif k.dim() == 1:
        k = k.view(-1, 1)
    else:
        print("Error: Dimension Error: expH")
        raise ValueError

    inpro = -th.sqrt(th.sum(y**2, dim=-1).view(-1, 1) * k + 1)
    cosh_inpro = th.arccosh(th.maximum(-inpro, th.ones(inpro.shape)))
    return cosh_inpro / sinh(cosh_inpro) * y



def MAdd(x, y, k=1): # Mobius Addition on Poincare Model
    # return x \oplus y

    if isinstance(k, float) or isinstance(k, int):
        if x.dim() == 2:
            k = (th.ones(x.shape[0])*k).view(-1,1)
        elif y.dim() == 2:
            k = (th.ones(y.shape[0])*k).view(-1,1)
    elif isinstance(k, list):
        k = th.tensor(k).view(-1, 1)
    elif type(k) == th.Tensor:
        if k.dim() == 1:
            k = k.view(-1, 1)
        elif k.dim() == 2:
            if k.shape[1] == 1:
                pass
            else:
                print("Error: Dimension Error: MAdd, k.shape = {}".format(k.shape))
                raise ValueError
        else:
            print("Error: Dimension Error: MAdd, k.shape = {}".format(k.shape()))
            raise ValueError
    else:
        print("Error: Dimension Error: MAdd")
        raise ValueError
    
    k = k.to(dtype)
    if x.dim() == 1 and y.dim() == 1:
        inpro = th.sum(x*y)
        nx = th.norm(x**2, p=1)
        ny = th.norm(y**2, p=1)
        return ((1 + 2 * k * inpro + k * ny)* x + (1-k*nx) * y) / (1 + 2 * k * inpro + k**2 * nx * ny)

    inpro = th.sum(x*y, dim=-1).view(-1, 1)
    nx = th.sum(x**2, dim=-1).view(-1, 1) # L1 normの二乗
    ny = th.sum(y**2, dim=-1).view(-1, 1)
    return ((1 + 2 * k * inpro + k * ny)* x + (1-k*nx) * y) / (1 + 2 * k * inpro + k**2 * nx * ny)

def MGyr(x, y, z, k=1): # Mobius Gyrator on Poincare Model
    # return gyr[x,y]z
    # x, y, z are in Poincare Model
    if isinstance(k, float) or isinstance(k, int):
        if x.dim() == 2:
            k = (th.ones(x.shape[0])*k).view(-1,1)
    elif isinstance(k, list):
        k = th.tensor(k).view(-1, 1)
    elif k.dim() == 1:
        k = k.view(-1, 1)
    else:
        print("Error: Dimension Error: MGyr")
        raise ValueError

    if x.dim() == 1:
        if y.dim() != 1 or z.dim() != 1:
            print("Error: Dimension Error: MGyr")
            raise ValueError
        xy = th.sum(x*y)
        yz = th.sum(y*z)
        xz = th.sum(x*z)
        nx = th.sum(x**2)
        ny = th.sum(y**2)
    else: # dim = 2
        if  x.dim != 2 or y.dim() != 2 or z.dim() != 2:
            print("Error: Dimension Error: MGyr")
            raise ValueError
        xy = th.sum(x*y, dim=1).view(-1, 1)
        yz = th.sum(y*z, dim=1).view(-1, 1)
        xz = th.sum(x*z, dim=1).view(-1, 1)
        nx = th.sum(x**2, dim=1).view(-1, 1)
        ny = th.sum(y**2, dim=1).view(-1, 1)
    
    return z - 2*k*( (k*xz*ny - yz*(1+2*k*xy))*x + (k*yz*nx + xz)*y ) / (1 + 2*k*xy + k**2*nx*ny)
    
def MMulti(x, r, k=1): # Mobius scalar multiplication on Poincare Model
    # x is in Poincare Model, r is scalar
    # k must either be an int, a float, a list or a torch.Tensor with dim == 1
    # return r \odot x
    if isinstance(k, float) or isinstance(k, int):
        k = (th.ones(x.shape[0])*k).view(-1,1)
    elif isinstance(k, list):
        k = th.tensor(k).view(-1, 1)
    elif k.dim() == 1:
        k = k.view(-1, 1)
    else:
        print("Error: Dimension Error: MMulti")
        raise ValueError
    
    k = k.to(dtype)
    sqrtk = th.sqrt(k)
    if x.dim() == 1:
        l2x = th.norm(x, p=2)
        return x / sqrtk * arctanh(r * tanh(sqrtk*l2x)) / th.maximum(l2x, th.ones(1, dtype=dtype)*1e-6)
    
    if isinstance(r, float) or isinstance(r, int):
        r = (th.ones(x.shape[0])*r).view(-1,1)
    elif r.dim() == 1:
        r = r.view(-1, 1)
    l2x = th.norm(x, p=2, dim=1).view(-1, 1)
    #return x / sqrtk * tanh(r * arctanh(sqrtk*l2x)) / th.maximum(l2x, th.ones(l2x.shape, dtype=dtype)*1e-9)
    coef = th.pow( (1-sqrtk * l2x) / (1 + sqrtk * l2x), r)
    return x / sqrtk * (1 - 2 * coef / (1 + coef)) / th.maximum(l2x, th.ones(l2x.shape, dtype=dtype)*1e-9)

def distP(X, Y, k=1): # Poicare Modelでの距離。X, YはPoincare Modelの座標
    #return 2 * arctanh(th.norm(-MAdd(ExpP0(X), ExpP0(Y)), dim=1))

    if k == 1:
        if X.dim() == 1:
            return 2 * arcsinh(th.sqrt((th.sum((X-Y)**2)) / th.maximum((1- th.sum(X**2)) * (1 - th.sum(Y**2)), th.ones(1)*1e-6)))

        return 2 * arcsinh(th.sqrt((th.sum((X-Y)**2, dim=1)) / th.maximum((1 - th.sum(X**2, dim=1))*(1 - th.sum(Y**2, dim=1)), th.ones(X.shape[0], dtype=dtype)*1e-6)))
    

    if isinstance(k, float) or isinstance(k, int):
        k = (th.ones(X.shape[0])*k).view(-1,1)
    elif isinstance(k, list):
        k = th.tensor(k).view(-1, 1)
    elif k.dim() == 1:
        k = k.view(-1, 1)
    else:
        print("Error: Dimension Error: distP")
        raise ValueError
    k = k.to(dtype)
    sqrtk = th.sqrt(k)

    z = MAdd(-X, Y, k)
    nz = th.norm(z, p=2, dim=1).view(-1, 1)
    return 2 / sqrtk * arctanh(sqrtk * nz)

def distH(X, Y, k=1): # 双曲面モデルでの距離
    if isinstance(k, float) or isinstance(k, int) or isinstance(k, list):
        sqrtk = th.sqrt(th.tensor(k, dtype=dtype))
    elif isinstance(k, th.Tensor):
        sqrtk = th.sqrt(k)
    if X.dim() == 1:
        xy0 = th.sqrt((th.sum(X*X) + 1/k) * (th.sum(Y*Y) + 1/k)).detach()
        Z = -sqrtk*th.sum(X*Y) + sqrtk*xy0
        return arccosh(th.maximum(Z, th.ones(Z.shape))) / sqrtk
        # return arccosh(Z + 1e-6) / k
    #xy0 = th.sqrt((th.sum(X*X, dim=1) + 1/k**2) * (th.sum(Y*Y, dim=1) + 1/k**2)).detach()
    xy0 = th.sqrt((th.sum(X*X, dim=1) + 1/k) * (th.sum(Y*Y, dim=1) + 1/k))
    Z = -k*th.sum(X*Y, dim=1) + k*xy0
    return arccosh(th.maximum(Z, th.ones(Z.size()))) / sqrtk

def H2P(X): # 双曲面モデルの座標をPoincare Modelの座標に変換
    if X.dim() == 1:
        return X / (1 + th.sqrt(1 + th.sum(X**2)))
    return X / (1 + th.sqrt(1 + th.sum(X**2, dim=1)).view(-1, 1))

def P2H(X): # Poincare Modelの座標を双曲面モデルの座標に変換
    if X.dim() == 1:
        return X / (1 - th.sqrt(1 - th.sum(X**2))) * 2
    return X / (1 - th.sqrt(1 - th.sum(X**2, dim=1)).view(-1, 1)) * 2

def P2B(X): #Poincare Model to Beltrami-Klein Model
    x = th.sum(X**2, dim=-1)
    x = th.maximum(x, th.ones(x.shape)*1e-6)
    y = 2 * th.sqrt(x)/ (1 + x)
    return X * (y/th.sqrt(x)).unsqueeze(-1).expand_as(X)

def B2P(X): #Beltrami-Klein Model to Poincare Model
    x = th.sqrt(th.sum(X**2, dim=-1))
    x = th.maximum(x, th.ones(x.shape)*1e-6)
    y1 = th.sqrt(1 + x)
    y2 = th.sqrt(1 - x)
    y = (y1 - y2) / (y1 + y2)
    return X * (y/x).unsqueeze(-1).expand_as(X)

def gyromidpoint_viaB(X_list): # Beltrami-Klein gyromidpoint]
    if isinstance(X_list, list):
        X_list = th.stack(X_list)
    X_list = P2B(X_list)
    gamma = 1 / th.sqrt(1 - th.sum(X_list**2, dim=-1))
    denom = th.sum(gamma, dim=0)
    coeff = (gamma / denom).unsqueeze(-1).expand_as(X_list)
    X = th.sum(coeff * X_list, dim=0)
    return B2P(X)

def gyromidpoint(x_list, a): # Poincare gyromidpoint
    # x_list: iterator of torch.Tensor
    # a: float
    if isinstance(a, list):
        N = len(a)
        a = th.tensor(a)
    elif isinstance(a, th.tensor):
        N = a.shape[0]
    else:
        print("Error: Type Error: gyromidpoint")
        raise ValueError
    if isinstance(x_list, list):
        x_list = th.stack(x_list)
        assert x_list.shape[0] == N
    elif isinstance(x_list, th.tensor):
        pass
    else:
        print("Error: Type Error: gyromidpoint")
        raise ValueError
    dim = x_list.shape[-1]
    #print(a.shape, x_list.shape)
    x2 = th.sum(x_list ** 2, dim=-1)
    gamma = 2 / (1-th.maximum(x2, th.ones(x2.shape)*1e-6)) # [N, ...]
    if x_list.dim() == 3: #(N, bsz, dim)
        a = a.unsqueeze(1).expand_as(x2) # [N, ...]
    elif x_list.dim() == 4: #(N, bsz, channel, dim)
        a = a.unsqueeze(1).unsqueeze(1).expand_as(x2) # [N, ...]
    denom = th.sum(th.abs(a) * (gamma-1), dim=0) # [...]
    #print(a.shape, gamma.shape, denom.shape)
    coeff = (a * gamma / denom).unsqueeze(-1).expand_as(x_list) # [N, ..., dim]
    #print(coeff.shape)
    x = th.sum(coeff * x_list, dim=0) # [..., dim]
    #print(x.shape)
    return MMulti(x, 1/2)

"""
x = expP0(th.randn((5,4)))
y = expP0(th.randn((5,4)))
zero = th.zeros(5,4)
z = gyromidpoint([x, y], [1, 1])
w = gyromidpoint_viaB([x, y])
print(distP(z, w))

"""
