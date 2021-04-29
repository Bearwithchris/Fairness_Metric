# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:19:55 2021

@author: SUTD
"""
import numpy as np
from scipy.stats import wasserstein_distance
import time

def fairness_discrepancy(data, n_classes, norm=0):
    """
    computes fairness discrepancy metric for single or multi-attribute
    this metric computes L2, L1, AND KL-total variation distance
    """
    unique, freq = np.unique(data, return_counts=True)
    props = freq / len(data) #Proportion of data that belongs to that data
    # print (freq)
    truth = 1./n_classes


    # L2 and L1=================================================================================================
    l2_fair_d = np.sqrt(((props - truth)**2).sum())/n_classes
    l1_fair_d = abs(props - truth).sum()/n_classes

    # q = props, p = truth
    # kl_fair_d = (props * (np.log(props) - np.log(truth))).sum()

    #Cross entropy
    p=np.ones(n_classes)/n_classes    
    # ce=cross_entropy(p,props,n_classes)-cross_entropy(p,p,n_classes)
    
    #information specificity=====================================================================================
    rank=np.linspace(1,n_classes-1,n_classes-1)
    rank[::-1].sort() #Descending order
    perc=np.array([i/np.sum(rank) for i in rank])
    
    #Create array to populate proportions
    props2=np.zeros(n_classes)
    props2[unique]=props
                  
    props2[::-1].sort()
    alpha=props2[1:]
    specificity=abs(props2[0]-np.sum(alpha*perc))
    info_spec=(l1_fair_d+specificity)/2
    
    #Wasstertein Distance
    wd=wasserstein_distance(props,np.ones(len(props))*truth)
    
    if norm==0:
        return l2_fair_d, l1_fair_d,info_spec,specificity,wd
        # return l2_fair_d, l1_fair_d,info_spec,specificity
    else:
        return l2_fair_d/metric_max(n_classes,"l2"), l1_fair_d/metric_max(n_classes,"l1"),info_spec/metric_max(n_classes,"is"),specificity,wd/metric_max(n_classes,"wd")
       # return l2_fair_d/metric_max(n_classes,"l2"), l1_fair_d/metric_max(n_classes,"l1"),info_spec/metric_max(n_classes,"is"),specificity

def metric_max(n_classes,Mtype):
    Pref=np.ones(n_classes)/n_classes #Reference attribute
    Pep=np.zeros(n_classes)
    Pep[0]=1
    if Mtype=="l1":
        fair_d = abs(Pep - Pref).sum()/n_classes
    elif Mtype=="l2":
        fair_d = np.sqrt(((Pep - Pref)**2).sum())/n_classes
    elif Mtype=="is":
        #L1
        l1_fair_d = abs(Pep - Pref).sum()/n_classes
        #Specificity
        rank=np.linspace(1,n_classes-1,n_classes-1)
        rank[::-1].sort() #Descending order
        perc=np.array([i/np.sum(rank) for i in rank])
        
        alpha=Pep[1:]
        specificity=abs(Pep[0]-np.sum(alpha*perc))
        fair_d=(l1_fair_d+specificity)/2
    elif Mtype=="wd":
        fair_d=wasserstein_distance(Pep,Pref)
    else:
        fair_d=0
    return fair_d


metric="wd"
print(metric_max(2,metric))
print(metric_max(4,metric))
print(metric_max(8,metric))
print(metric_max(16,metric))
