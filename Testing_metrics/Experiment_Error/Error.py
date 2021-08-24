# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:19:55 2021

@author: SUTD
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import time
import os

def dist_to_csv():
    n_class_array=[2,4,8,16]
    for n_class in n_class_array:
        #Ideal data
        ideal_dist=np.load("./A_t_{}/ideal_dist.npz".format(n_class))['x']
        ideal_dist=pd.DataFrame(ideal_dist)
        ideal_dist.to_csv("psuedo_random_dist_{}.csv".format(n_class))
        #Predicted
        pred_dist=np.load("./A_t_{}/pred_dist.npz".format(n_class))['x']
        pred_dist=pd.DataFrame(pred_dist)
        pred_dist.to_csv("predicted_dist_{}.csv".format(n_class))


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

def measuring_scores_single(n_class,prefix="./",norm=0,debug_mode=0):
    #open up the data 

    error=np.zeros(1000)
    
    #Log files
    if debug_mode==0:
        if norm==0:
            file=open(os.path.join(prefix,"scores_report_at_{}.csv".format(n_class)),'w')
        else:
            file=open(os.path.join(prefix,"scores_report_at_{}_normalised.csv").format(n_class),'w')
            
    #Load pre-classified data
    ideal_dist=np.load(os.path.join(prefix,"ideal_dist.npz".format(n_class)))['x']
    pred_dist=np.load(os.path.join(prefix,"pred_dist.npz".format(n_class)))['x']
    
    #Prepare CSV header
    if debug_mode==0:
        file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format("pred-L2","pred-L1","pred-IS","pred-SP","pred-WD","actual-L2","actual-L1","actual-IS","actual-SP","actual-WD","delta-L2","delta-L1","delta-IS","delta-SP","delta-WD"))
        
    #Measure FD score error
    for i in range(len(ideal_dist)):
        
        #Unnormalised
        if norm==0:     
            f=fairness_discrepancy(pred_dist[i],n_class,norm=0)
            fr=fairness_discrepancy(ideal_dist[i],n_class,norm=0)
            delta=abs(np.asarray(fr)-np.asarray(f))
            if debug_mode==0:
                file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(f[0],f[1],f[2],f[3],f[4],fr[0],fr[1],fr[2],fr[3],fr[4],delta[0],delta[1],delta[2],delta[3],delta[4]))
        #Normalised
        else:
            f=fairness_discrepancy(pred_dist[i],n_class,norm=1)
            fr=fairness_discrepancy(ideal_dist[i],n_class,norm=1)
            delta=abs(np.asarray(fr)-np.asarray(f))
            if debug_mode==0:
                file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(f[0],f[1],f[2],f[3],f[4],fr[0],fr[1],fr[2],fr[3],fr[4],delta[0],delta[1],delta[2],delta[3],delta[4]))
            
    if debug_mode==0:
        file.close()

def measuring_scores(norm=0,debug_mode=0):
    #open up the data 
    n_class_array=[2,4,8,16]
    for n_class in n_class_array:
        prefix="./A_t_{}".format(n_class)
        error=np.zeros(1000)
        
        #Log files
        if debug_mode==0:
            if norm==0:
                file=open(os.path.join("scores_report_at_{}.csv".format(n_class)),'w')
            else:
                file=open(os.path.join(prefix,"scores_report_at_{}_normalised.csv").format(n_class),'w')
                
        #Load pre-classified data
        ideal_dist=np.load(os.path.join(prefix,"ideal_dist.npz".format(n_class)))['x']
        pred_dist=np.load(os.path.join(prefix,"pred_dist.npz".format(n_class)))['x']
        
        #Prepare CSV header
        if debug_mode==0:
            file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format("pred-L2","pred-L1","pred-IS","pred-SP","pred-WD","actual-L2","actual-L1","actual-IS","actual-SP","actual-WD","delta-L2","delta-L1","delta-IS","delta-SP","delta-WD"))
            
        #Measure FD score error
        for i in range(len(ideal_dist)):
            
            #Unnormalised
            if norm==0:     
                f=fairness_discrepancy(pred_dist[i],n_class,norm=0)
                fr=fairness_discrepancy(ideal_dist[i],n_class,norm=0)
                delta=abs(np.asarray(fr)-np.asarray(f))
                if debug_mode==0:
                    file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(f[0],f[1],f[2],f[3],f[4],fr[0],fr[1],fr[2],fr[3],fr[4],delta[0],delta[1],delta[2],delta[3],delta[4]))
            #Normalised
            else:
                f=fairness_discrepancy(pred_dist[i],n_class,norm=1)
                fr=fairness_discrepancy(ideal_dist[i],n_class,norm=1)
                delta=abs(np.asarray(fr)-np.asarray(f))
                if debug_mode==0:
                    file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(f[0],f[1],f[2],f[3],f[4],fr[0],fr[1],fr[2],fr[3],fr[4],delta[0],delta[1],delta[2],delta[3],delta[4]))
                
        if debug_mode==0:
            file.close()
   

def measuring_ub_error(metric="L2",norm=1,debug_mode=0):
    #open up the data 
    dict_metric={'L2':0, 'L1':1,'Is':2,'Sp':3}
    n_class_array=[2,4,8,16]
    rounddp=5
    for n_class in n_class_array:
        error=np.zeros(1000)
        delta_f_array=np.zeros(1000)
        
        #Log files
        if debug_mode==0:
            if norm==0:
                file=open("./{}_report_at_{}.csv".format(metric,n_class),'w')
            else:
                file=open("./{}_report_at_{}_normalised.csv".format(metric,n_class),'w')
                
        #Load pre-classified data
        ideal_dist=np.load("./A_t_{}/ideal_dist.npz".format(n_class))['x']
        pred_dist=np.load("./A_t_{}/pred_dist.npz".format(n_class))['x']
        
        #Prepare CSV header
        if debug_mode==0:
            file.write("{},{},{},{},{}\n".format("delta_p","f_star","f","delta_f","acceptable_error"))
            
        #Measure FD score error
        for i in range(len(ideal_dist)):
            def Sp_unnorm():
                rank=np.linspace(1,n_class-1,n_class-1)
                rank[::-1].sort() #Descending order
                perc=np.array([i/np.sum(rank) for i in rank])
                ideal_dist[i]=np.sort(ideal_dist[i])[::-1]
                pred_dist[i]=np.sort(pred_dist[i])[::-1]
                delta_p=abs(ideal_dist[i][0]-pred_dist[i][0])+abs((perc*pred_dist[i][1:]).sum()-(perc*ideal_dist[i][1:]).sum())  
                return delta_p
            def L1_unnorm():
                delta_p=abs(ideal_dist[i]-pred_dist[i]).sum()
                return delta_p
            
            #Unnormalised
            if norm==0:
                if metric=="L1":
                    # delta_p=abs(ideal_dist[i]-pred_dist[i]).sum()
                    delta_p=L1_unnorm()
                elif metric=="L2": #L2
                    delta_p=np.sqrt(((ideal_dist[i] - pred_dist[i])**2).sum())
                elif metric=="Sp":
                    # rank=np.linspace(1,n_class-1,n_class-1)
                    # rank[::-1].sort() #Descending order
                    # perc=np.array([i/np.sum(rank) for i in rank])
                    # ideal_dist[i]=np.sort(ideal_dist[i])[::-1]
                    # pred_dist[i]=np.sort(pred_dist[i])[::-1]
                    # delta_p=abs(ideal_dist[i][0]-pred_dist[i][0])+abs((perc*pred_dist[i][1:]).sum()-(perc*ideal_dist[i][1:]).sum())
                    delta_p=Sp_unnorm()
                elif metric=="Is":
                    alpha=0.5
                    L1_delta_p=L1_unnorm()
                    Sp_delta_p=Sp_unnorm()
                    delta_p=alpha*abs(L1_delta_p)+(1-alpha)*abs(Sp_delta_p)
                    # Sp_delta_p=
                    
                    # validate=abs((ideal_dist[i][0]-pred_dist[i][0])+((perc*pred_dist[i][1:]).sum()-(perc*ideal_dist[i][1:]))).sum()
    
                    # delta_p=abs(fairness_discrepancy(ideal_dist[i],n_class,norm=0)[dict_metric[metric]] - fairness_discrepancy(pred_dist[i],n_class,norm=0)[dict_metric[metric]] )
                else:
                    print ("Error in metric choice!")
                    break
                            
                #Calculare the error componenets
                delta_p=np.around(delta_p,rounddp)
                f_star=fairness_discrepancy(ideal_dist[i],n_class,norm=0)
                f=fairness_discrepancy(pred_dist[i],n_class,norm=0)
                delta_f=np.around(abs(f_star[dict_metric[metric]]-f[dict_metric[metric]]),rounddp)
            
            #Normalised
            else:
                nfactor=metric_max(n_class,metric)
                if metric=="L1":
                    delta_p=(abs(ideal_dist[i]-pred_dist[i]).sum())/nfactor
                elif metric=="L2": #L2
                     delta_p=(np.sqrt(((ideal_dist[i] - pred_dist[i])**2).sum()))/nfactor
                elif metric=="Sp":
                    rank=np.linspace(1,n_class-1,n_class-1)
                    rank[::-1].sort() #Descending order
                    perc=np.array([i/np.sum(rank) for i in rank])
                    ideal_dist[i]=np.sort(ideal_dist[i])[::-1]
                    pred_dist[i]=np.sort(pred_dist[i])[::-1]
                    delta_p=abs(ideal_dist[i][0]-pred_dist[i][0])+abs(perc*pred_dist[i][1:]-perc*ideal_dist[i][1:]).sum()
                    # delta_p=abs(fairness_discrepancy(ideal_dist[i],n_class,norm=0)[dict_metric[metric]] - fairness_discrepancy(pred_dist[i],n_class,norm=0)[dict_metric[metric]] )
                elif metric=="Is":
                    alpha=0.5
                    L1_delta_p=L1_unnorm()
                    Sp_delta_p=Sp_unnorm()
                    delta_p=(alpha*abs(L1_delta_p)+(1-alpha)*abs(Sp_delta_p))/nfactor
                    # Sp_delta_p=
                else:
                    print ("Error in metric choice!")
                    break
                     
                #Calculare the error componenets
                delta_p=np.around(delta_p,rounddp)
                f_star=fairness_discrepancy(ideal_dist[i],n_class,norm=1)
                f=fairness_discrepancy(pred_dist[i],n_class,norm=1)
                delta_f=np.around(abs(f_star[dict_metric[metric]]-f[dict_metric[metric]]),rounddp)
                
                
                
            
            error[i]=delta_p
            delta_f_array[i]=delta_f
            
            #Error range acceptable?
            if delta_f<=delta_p:
            # if delta_f<=delta_p:
                acceptable=1
                # print ("Accept")
            else:
                acceptable=0
                print ("Reject")
            if debug_mode==0:    
                file.write("{},{},{},{},{}\n".format(delta_p,f_star[1],f[1],delta_f,acceptable))
        if norm==0:
            print ("Mean/SD error for attribute {} i.e. mean_delta_p={} and SD_delta_p={}".format(n_class,np.mean(error),np.std(error)))
            print ("Mean/SD Delta_f for attribute {} ={} and SD_delta_f={}".format(n_class,np.mean(delta_f_array),np.std(delta_f_array)))
        else:
            print ("Mean/SD Norm error for attribute {} i.e. mean_delta_p={} and SD_delta_p={}".format(n_class,np.mean(error),np.std(error)))
            print ("Mean/SD Norm Delta_f for attribute {} ={} and SD_delta_f={}".format(n_class,np.mean(delta_f_array),np.std(delta_f_array)))
        if debug_mode==0:
            file.close()

#Run upper bound
metrics_array=["L2","L1","Is","Sp"]
for m in metrics_array:
    print (m)
    measuring_ub_error(metric=m,norm=1,debug_mode=1)
    print ("========================================================================")
#Run scores in norm mode
# measuring_scores(norm=1)
#Run scores in single at
# k=4
# perc=90
# at="31_39"
# prefix="./bias_one/{}_At_{}/{}".format(k,at,perc)
# measuring_scores_single(4,prefix,norm=1)