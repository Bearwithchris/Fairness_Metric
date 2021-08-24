# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:46:17 2021

@author: Chris
"""
import numpy as np
import copy 
# import numba
import time
from scipy.stats import wasserstein_distance

np.random.seed(20)
#Random array generator
def random(k):
    dist=np.random.rand(1,k)
    dist=dist/dist.sum()
    return (dist[0])

#Intialising the label ground truth with initial ibias
def populate_Lgt(Tperc,k,Nsamples=3000,skew=0):
    bias_sample_count=round(Tperc*Nsamples)
    bias_samples=np.zeros(bias_sample_count)
    if skew==0:
        #Uniform remaining
        remaining_sample_count=round((Nsamples*(1-Tperc))/(k-1))
        remaining_sample_count_array=np.ones(k-1)*remaining_sample_count
    else:
        #Skewed remaining dist
        remaining_sample_count=round((Nsamples*(1-Tperc)))
        remaining_sample_count_array=np.arange(k-1,0,-1)/sum(np.arange(k-1,0,-1))*remaining_sample_count
        
        
    for i in range (0,k-1):
        __samples=np.ones(int(remaining_sample_count_array[i]))*(i+1)
        bias_samples=np.concatenate([bias_samples,__samples])
    return bias_samples


#Generate the predicted labels, simulating the classifier of a given accuracy Tpre
def gen_pred_labels(k,ibias,cAcc,Lgt,Nsamples,bias_flip=0):
    # start=time.time()
    # Lgt=populate_Lgt(ibias,k,Nsamples) #Label ground truth
    Lpred=copy.deepcopy(Lgt)
    
    #break statement when Tperc is achieved
    while (sum(Lgt==Lpred)/Nsamples)!=cAcc:
        positiveLoc=np.where((Lgt==Lpred)==1)
        if bias_flip==0:
            flip_index=np.random.choice(np.where((Lgt==Lpred)==1)[0])
            selection=np.delete(np.arange(0.0,4.0,1.0),int(Lpred[flip_index]))
            Lpred[flip_index]=np.random.choice(selection)
        else:
            # bias=np.array([0.4,0.2,0.2,0.2]) #Larger the prob implies, lower the accuracy of that At
            select_flip=np.random.choice(np.arange(0.0,4.0,1.0)) #Selected attribute to flip
            try:
                flip_index=np.random.choice(np.where(((Lgt==Lpred)==1) & (Lgt==select_flip) )[0])           
                # flip_index=np.random.choice(np.where((Lgt==Lpred)==1)[0])
                # bias2=np.flip(bias)
                Lpred[flip_index]=np.random.choice(np.arange(0.0,4.0,1.0))
            except:
                pass
        # print ("Current acc is: {}".format(sum(Lgt==Lpred)/Nsamples))
    
    # print ("time taken: "+str(time.time()-start))
    return Lpred

def L2P(Lpred):
    unique, freq = np.unique(Lpred, return_counts=True)
    props = freq / len(Lpred) #Proportion of data that belongs to that data
    return props


def fairness_discrepancy(props, n_classes, norm=0):
    """
    computes fairness discrepancy metric for single or multi-attribute
    this metric computes L2, L1, AND KL-total variation distance
    """
    # unique, freq = np.unique(data, return_counts=True)
    # props = freq / len(data) #Proportion of data that belongs to that data
    
    # #------------------Modification to correct the zero support problem------------------------------------------------
    # temp=np.zeros(n_classes)
    # temp[unique]=props
    # props=temp
    # #------------------------------------------------------------------------------
    
    # print (freq)
    truth = 1./n_classes


    # L2 and L1=================================================================================================
    #(Remove Normalisation)
    l2_fair_d = np.sqrt(((props - truth)**2).sum())
    l1_fair_d = abs(props - truth).sum()

    # q = props, p = truth
    # kl_fair_d = (props * (np.log(props) - np.log(truth))).sum()

    #Cross entropy
    p=np.ones(n_classes)   
    # ce=cross_entropy(p,props,n_classes)-cross_entropy(p,p,n_classes)
    
    #information specificity=====================================================================================
    rank=np.linspace(1,n_classes-1,n_classes-1)
    rank[::-1].sort() #Descending order
    perc=np.array([i/np.sum(rank) for i in rank])
    
                  
    props[::-1].sort()
    alpha=props[1:]
    specificity=abs(props[0]-np.sum(alpha*perc))
    info_spec=(l1_fair_d+specificity)/2
    
    #Wasstertein Distance
    wd=wasserstein_distance(props,np.ones(len(props))*truth)
    
    #Wassertein Specificity
    wds=(wd+specificity)/2
    if norm==0:
        return l2_fair_d, l1_fair_d,info_spec,specificity,wd,wds
        # return l2_fair_d, l1_fair_d,info_spec,specificity
    else:
        return l2_fair_d/metric_max(n_classes,"L2"), l1_fair_d/metric_max(n_classes,"L1"),info_spec/metric_max(n_classes,"Is"),specificity,wd/metric_max(n_classes,"Wd")
       # return l2_fair_d/metric_max(n_classes,"l2"), l1_fair_d/metric_max(n_classes,"l1"),info_spec/metric_max(n_classes,"is"),specificity

def metric_max(n_classes,Mtype):
    Pref=np.ones(n_classes)/n_classes #Reference attribute
    Pep=np.zeros(n_classes)
    Pep[0]=1
    if Mtype=="L1":
        fair_d = abs(Pep - Pref).sum()
    elif Mtype=="L2":
        fair_d = np.sqrt(((Pep - Pref)**2).sum())
    elif Mtype=="Is":
        #L1
        l1_fair_d = abs(Pep - Pref).sum()
        #Specificity
        rank=np.linspace(1,n_classes-1,n_classes-1)
        rank[::-1].sort() #Descending order
        perc=np.array([i/np.sum(rank) for i in rank])
        
        alpha=Pep[1:]
        specificity=abs(Pep[0]-np.sum(alpha*perc))
        fair_d=(l1_fair_d+specificity)/2
    elif Mtype=="Wd":
        fair_d=wasserstein_distance(Pep,Pref)
    elif Mtype=="Wds":
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

def ideal_f(k,bias,n=1):
    Lgt=populate_Lgt(bias,k,Nsamples=3000)
    p=L2P(Lgt)
    # p=np.array([0.9,0.2,0.4,0.2])
    f=fairness_discrepancy(p,k,norm=n)
    print ("Ideal f at {} is L1={} L2={} Is={} Sp={}".format(bias,f[0],f[1],f[2],f[3]))
    return (f[0],f[1],f[2],f[3])
    
if __name__=="__main__":
    #Init parameters 
    k=4 #Cardinality of attributes (constant)
    ibias=0.7 #initial bias (constant)
    Tperc=0.7 #Target percentage (independent variable)
    Nsamples=3000 #Number of samples (constant)
    
    ideal_f(4,0.9,1)
    
    # Lgt=populate_Lgt(ibias,k,Nsamples)
    # Lpred=gen_pred_labels(k,ibias,Tperc,Lgt,Nsamples)

# unique, freq = np.unique(Lpred, return_counts=True)
# props = freq / len(Lpred) #Proportion of data that belongs to that data