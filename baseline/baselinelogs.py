# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:37:30 2021

@author: Chris
"""
import numpy as np
import copy 
from scipy.stats import wasserstein_distance
import time


def dist2(count):
    if count==2 or count==4:
        step=1
    elif count==8:
        step=0.5
    else: #count==16
        step=0.25
    #Make perfectly bias
    dist_base=np.zeros(count)
    dist_base[0]=100
    
    #Target
    even=100/count
    target=np.ones(count)*even
    
    #Loop parameters
    dist_array=[]
    index=1
    while (np.array_equal(dist_base,target)!=True):
        if (dist_base[index]!=target[index]):
            #Transfer from index 0
            dist_base[0]=dist_base[0]-step
            dist_base[index]=dist_base[index]+step
        else:
            index+=1
        array=copy.deepcopy(dist_base) 
        array=array/100
        dist_array.append(array)
        
    return dist_array

def fairness_discrepancy(props, n_classes, norm=0):
    """
    computes fairness discrepancy metric for single or multi-attribute
    this metric computes L2, L1, AND KL-total variation distance
    """
    # unique, freq = np.unique(data, return_counts=True)
    # props = freq / len(data) #Proportion of data that belongs to that data
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
    props2=copy.deepcopy(props)
 
                  
    props2[::-1].sort()
    alpha=props2[1:]
    specificity=abs(props2[0]-np.sum(alpha*perc))
    info_spec=(l1_fair_d+specificity)/2
    
    #Wasstertein Distance
    wd=wasserstein_distance(props2,np.ones(len(props2))*truth)
    
    #Wassertein Specificity
    wds=(wd+specificity)/2
    if norm==0:
        return l2_fair_d, l1_fair_d,info_spec,specificity,wd,wds
        # return l2_fair_d, l1_fair_d,info_spec,specificity
    else:
        return l2_fair_d/metric_max(n_classes,"l2"), l1_fair_d/metric_max(n_classes,"l1"),info_spec/metric_max(n_classes,"is"),specificity,wd/metric_max(n_classes,"wd"),wds/metric_max(n_classes,"wds")
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
    elif Mtype=="wds":
        #Specificity
        rank=np.linspace(1,n_classes-1,n_classes-1)
        rank[::-1].sort() #Descending order
        perc=np.array([i/np.sum(rank) for i in rank])
        alpha=Pep[1:]
        specificity=abs(Pep[0]-np.sum(alpha*perc))
        #Wassertein dist
        ws=wasserstein_distance(Pep,Pref)
        
        fair_d=(ws+specificity)/2    
    else:
        fair_d=0
    return fair_d

attributes=16
distArray=dist2(attributes)
f=open("../logs/BaseLine.txt",'a' )
count=0
f.write("classs index l2 l1 IS Specificity wd wds \n")
for i in distArray:
    l2,l1,IS,S,wd,wds=fairness_discrepancy(i,attributes,norm=1)
    f.write('{} {} {} {} {} {} {} {} \n'.format(attributes,count, l2, l1, IS,S,wd,wds))
    count+=1
f.close()