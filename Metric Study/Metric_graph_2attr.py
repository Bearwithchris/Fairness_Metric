# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:29:33 2021

@author: Chris
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import entropy

#Genearate Test Samples
at1=np.arange(0.01,1,0.01)
at2=1-at1
y=np.stack((at1,at2),1)

#ref
n_classes=2
truth = 1./n_classes
# truth=np.ones_like(y)*truth

def L2(attr,truth):
    L2_dist=np.zeros(len(attr))
    for i in range(len(attr)):    
        l2_fair_d = np.sqrt(((attr[i] - truth)**2).sum())
        L2_dist[i]=l2_fair_d
    return L2_dist

def L1(attr,truth):
    L1_dist=np.zeros(len(attr))
    for i in range(len(attr)):    
        l1_fair_d = abs(attr[i] - truth).sum()
        L1_dist[i]=l1_fair_d
    return L1_dist

def KL(attr,truth):
    KL_dist=np.zeros(len(attr))
    for i in range(len(attr)):    
        KL_fair_d =(attr[i] * (np.log(attr[i]) - np.log(truth))).sum()
        KL_dist[i]=KL_fair_d
    return KL_dist

def JSD(attr, truth):
    JSD_dist=np.zeros(len(attr))
    for i in range(len(attr)):  
        _M = 0.5 * (attr[i] + truth)
        JSD_dist[i]=0.5 * (entropy(attr[i], _M) + entropy(np.ones_like(_M)*truth, _M))
    return JSD_dist

def chebyshev(attr,truth):
    chebyshev=np.zeros(len(attr))
    for i in range(len(attr)):    
        chebyshev_fair_d =distance.chebyshev(attr[i],truth)
        chebyshev[i]=chebyshev_fair_d
    return chebyshev
    
x=np.arange(1,len(at1)+1,1)
L2=L2(y,truth)
L1=L1(y,truth)
KL=KL(y,truth)
JSD=JSD(y,truth)
chebyshev=chebyshev(y,truth)

plt.plot(x,L2,label="L2")
plt.plot(x,L1,label="L1")
plt.plot(x,KL,label="KL")
plt.plot(x,JSD,label="JSD")
plt.plot(x,chebyshev,label="Chebyshev")
plt.legend()

