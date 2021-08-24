# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:23:19 2021

@author: Chris
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import utils

def plot_fvAcc_exp_single(metric,ibias):
    # metric="L1"
    k=4 #Cardinality of attributes (constant)
    # ibias=0.8 #initialise bias (constant)
    cAcc_array=[0.9,0.8,0.7]
    # cAcc=0.9 #classifier's accuracy (independent variable)
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)  
    
    data_array=np.zeros([100,3])
    for index in range(len(cAcc_array)):
        file=np.load("bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,cAcc_array[index],k))[metric]
        data_array[:,index]=file
        # ax.scatter(np.array([cAcc for i in range(100)]),file)
        ax.set_xticklabels(cAcc_array) 
        print ("Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric,k,cAcc_array[index],ibias,np.around(np.mean(file),4),np.around(np.std(file),4)) )
    
    plt.rcParams.update({'font.size': 20})
    plt.title(metric+" ,$f$ vs $Acc_{avg}$ at bias= "+str(ibias))
    ax.set_ylabel("$f$")
    ax.set_xlabel("$Acc_{avg}$")
    ax.boxplot(data_array)
    plt.savefig("f_vs_acc_at_bias_{}.png".format(ibias))

def plot_fvAcc_exp_multi(ibias,prefix="./"):
    metric_array=["L1","L2","Is","Sp"]
    k=4 #Cardinality of attributes (constant)
    # ibias=0.8 #initialise bias (constant)
    cAcc_array=[0.9,0.8,0.7,0.6,0.5]
    # cAcc_array=[0.9,0.8,0.7]
    # cAcc=0.9 #classifier's accuracy (independent variable)
    
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(144)  
    ax2 = fig.add_subplot(143)
    ax3 = fig.add_subplot(142)
    ax4 = fig.add_subplot(141)  
    
    data_array_L1=np.zeros([100,len(cAcc_array)])
    data_array_L2=np.zeros([100,len(cAcc_array)])
    data_array_Is=np.zeros([100,len(cAcc_array)])
    data_array_Sp=np.zeros([100,len(cAcc_array)])
    
    counter=0
    for index in cAcc_array:
        file_L1=np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[0]]
        file_L2=np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[1]]
        file_Is=np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[2]]
        file_Sp=np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[3]]
        
        
        data_array_L1[:,counter]=file_L1
        data_array_L2[:,counter]=file_L2
        data_array_Is[:,counter]=file_Is
        data_array_Sp[:,counter]=file_Sp        
        counter+=1
        
        try:
            print ("Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[0],k,index,ibias,np.around(np.mean(file_L1),4),np.around(np.std(file_L1),4)) )
            print ("Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[1],k,index,ibias,np.around(np.mean(file_L2),4),np.around(np.std(file_L2),4)) )
            print ("Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[2],k,index,ibias,np.around(np.mean(file_Is),4),np.around(np.std(file_Is),4)) )
            print ("Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[3],k,index,ibias,np.around(np.mean(file_Sp),4),np.around(np.std(file_Sp),4)) )
        except:
            print("error")
                
    plt.rcParams.update({'font.size': 14})        
    ax1.set_xticklabels(cAcc_array) 
    ax2.set_xticklabels(cAcc_array) 
    ax3.set_xticklabels(cAcc_array) 
    ax4.set_xticklabels(cAcc_array) 
    ax4.set_ylabel("$f$")
    
    ax4.boxplot(data_array_L1)
    ax3.boxplot(data_array_L2)
    ax2.boxplot(data_array_Is)
    ax1.boxplot(data_array_Sp)
    
    ax4.title.set_text('L1')
    ax3.title.set_text('L2')
    ax2.title.set_text('Is')
    ax1.title.set_text('$\Delta sp$')
    

    plt.savefig(os.path.join(prefix,"overall_f_vs_acc_at_bias_{}.png".format(ibias)))
    
def plot_fvAcc_exp_multi_deltaf(ibias,prefix="./"):
    metric_array=["L1","L2","Is","Sp"]
    k=4 #Cardinality of attributes (constant)
    # ibias=0.8 #initialise bias (constant)
    cAcc_array=[0.9,0.8,0.7,0.6,0.5]
    # cAcc=0.9 #classifier's accuracy (independent variable)
    
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(144)  
    ax2 = fig.add_subplot(143)
    ax3 = fig.add_subplot(142)
    ax4 = fig.add_subplot(141)  
    
    data_array_L1=np.zeros([100,len(cAcc_array)])
    data_array_L2=np.zeros([100,len(cAcc_array)])
    data_array_Is=np.zeros([100,len(cAcc_array)])
    data_array_Sp=np.zeros([100,len(cAcc_array)])
    
    counter=0
    for index in cAcc_array:
        fstar=utils.ideal_f(k,index,n=1)
        file_L1=abs(fstar[0]-np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[0]])
        file_L2=abs(fstar[0]-np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[1]])
        file_Is=abs(fstar[0]-np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[2]])
        file_Sp=abs(fstar[0]-np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[3]])
        
        
        data_array_L1[:,counter]=file_L1
        data_array_L2[:,counter]=file_L2
        data_array_Is[:,counter]=file_Is
        data_array_Sp[:,counter]=file_Sp        
        counter+=1
        
        try:
            print ("Delta f, Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[0],k,index,ibias,np.around(np.mean(file_L1),4),np.around(np.std(file_L1),4)) )
            print ("Delta f, Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[1],k,index,ibias,np.around(np.mean(file_L2),4),np.around(np.std(file_L2),4)) )
            print ("Delta f, Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[2],k,index,ibias,np.around(np.mean(file_Is),4),np.around(np.std(file_Is),4)) )
            print ("Delta f, Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[3],k,index,ibias,np.around(np.mean(file_Sp),4),np.around(np.std(file_Sp),4)) )
        except:
            print("error")
                
    plt.rcParams.update({'font.size': 14})        
    ax1.set_xticklabels(cAcc_array) 
    ax2.set_xticklabels(cAcc_array) 
    ax3.set_xticklabels(cAcc_array) 
    ax4.set_xticklabels(cAcc_array) 
    ax4.set_ylabel("$\Delta f$")
    
    ax4.boxplot(data_array_L1)
    ax3.boxplot(data_array_L2)
    ax2.boxplot(data_array_Is)
    ax1.boxplot(data_array_Sp)
    
    ax4.title.set_text('L1')
    ax3.title.set_text('L2')
    ax2.title.set_text('Is')
    ax1.title.set_text('$\Delta sp$')
    

    plt.savefig(os.path.join(prefix,"overall_Delta f_vs_acc_at_bias_{}.png".format(ibias)))


metric="L1"
ibias=0.7 #initialise bias (constant)
# plot_fvAcc_exp_single(metric,ibias)
prefix="./"
plot_fvAcc_exp_multi(ibias,prefix)
plot_fvAcc_exp_multi_deltaf(ibias,prefix)