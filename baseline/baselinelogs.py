# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:37:30 2021

@author: Chris
"""
import numpy as np
import copy 
from scipy.stats import wasserstein_distance
import time
import matplotlib.pyplot as plt


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


def plot(label_list,array_list):
    styles=['--o','--','-.',',',':','-']
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    for i in range(1,len(label_list)):
        ax.plot(array_list[0], array_list[i],styles[i-1],label=label_list[i],color='k')
        ax.legend( prop={'size': 20})
        # plt.plot(array_list[0],array_list[i],styles[i-1])
    # plt.show()
    return fig

def delta_f(label_list,array_list):
    array_list=np.array(array_list)
    deltaf=np.zeros(len(array_list))
    for i in range(1,len(array_list[0])):
        deltaf=deltaf+(array_list[:,i-1]-array_list[:,i])
    deltaf=deltaf/len(array_list[0])
    return deltaf[1:]
        
    
attributes_array=[2,4,8,16]
for attributes in attributes_array:
    distArray=dist2(attributes)
    f=open("../logs/BaseLine.txt",'a' )
    count=0
    f.write("classs index l2 l1 IS Specificity wd linear \n")
    array_list=[[],[],[],[],[],[],[]]
    array_list_attr=["Count","L2","L1","IS","Specificity","WD","Linear"]
    for i in distArray:
        linear=(len(distArray)-count)/len(distArray)
        l2,l1,IS,S,wd,wds=fairness_discrepancy(i,attributes,norm=1)
        f.write('{} {} {} {} {} {} {} {} \n'.format(attributes,count, l2, l1, IS,S,wd,linear))
        fairness_disc_scores=[count,l2,l1,IS,S,wd,linear]
        for j in range(len(array_list_attr)):
            array_list[j].append(fairness_disc_scores[j])
        
        count+=1
    f.close()
    fig=plot(array_list_attr,array_list)
    fig.savefig("../logs/attr_%i_baseline.pdf"%attributes)
    deltaf=delta_f(array_list_attr,array_list)
    f=open("../logs/BaseLine_delta_f.txt",'a' )
    f.write("Attributes: %i,L2 L1 IS Specificity WD Linear \n"%(attributes))
    f.write('{} {} {} {} {} {} \n'.format(deltaf[0], deltaf[1], deltaf[2],deltaf[3],deltaf[4],deltaf[5]))
    f.close()