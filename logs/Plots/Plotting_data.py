# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:36:19 2021

@author: Chris
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

palette={'L2':"0",'L1':"0.2",'IS':"0.4",'SP':"0.6","WD":'0.8'}
plt.rcParams.update({'font.size': 28})

'''
Classification Error Experiment Plots
'''
def Classification_error_AB_EP(k=2):  
    if k==2:
        L2=[0.9577,0.924,0.519,0.7199]
        L1=[0.9577,0.924,0.519,0.7199]
        IS=[0.9577,0.924,0.519,0.7199]
        SP=[0.9577,0.924,0.519,0.7199]
        WD=[0.9577,0.924,0.519,0.7199]
    else:     
        L2=[0.7537,0.8926,0.7459,0.7916,0.7364,0.6824,0.5576,0.6702]    
        L1=[0.7403,0.8911,0.7383,0.7859,0.7331,0.6726,0.5233,0.6562]
        IS=[0.7208,0.8844,0.7238,0.7721,0.7231,0.6556,0.4954,0.6385]
        SP=[0.7135,0.8819,0.7184,0.7669,0.7194,0.6493,0.4849,0.6318]
        WD=[0.7403,0.8911,0.7383,0.7859,0.7331,0.6726,0.5233,0.6562]
        
    barWidth = 1/7
    br1 = np.arange(len(L2))
    br2 = [x+0.025 + barWidth for x in br1]
    br3 = [x+0.025 + barWidth for x in br2]
    br4 = [x+0.025 + barWidth for x in br3]
    br5 = [x+0.025 + barWidth for x in br4]
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    
    
    
    ax.bar(br1, L2, color =palette['L2'], width = barWidth,
            edgecolor ='grey', label ='L2')
    ax.bar(br2, L1, color =palette['L1'], width = barWidth,
            edgecolor ='grey', label ='L1')
    ax.bar(br3, IS, color =palette['IS'], width = barWidth,
            edgecolor ='grey', label ='IS')
    ax.bar(br4, SP, color =palette['SP'], width = barWidth,
            edgecolor ='grey', label ='SP')
    ax.bar(br5, WD, color =palette['WD'], width = barWidth,
            edgecolor ='grey', label ='WD')
    
    #Labels
    # ax.set_xticklabels([r + barWidth for r in range(len(L2))],
    #         ["[0,1]", "[1,0]", "[0,1]", "[1,0]", "[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]","[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]"])
    if k==2:
        ax.set_xticks([x+0.3 for x in range(4)])
        ax.set_xticklabels(["[0,1]", "[1,0]", "[0,1]", "[1,0]"])
        ax.legend()
        ax2=ax.twiny()
        ax2.set_xticks([0.3,0.7])
        ax2.set_xticklabels(["Male-Female","Young-Old"])
    else:
        ax.set_xticks([x+0.3 for x in range(8)])
        ax.set_xticklabels(["[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]","[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]"])
        ax.legend()
        ax2=ax.twiny()
        ax2.set_xticks([0.3,0.7])
        ax2.set_xticklabels(["Male-Female,Black Hair","Young-Old,Smiling"])
    ax.legend( prop={'size': 18})
    
    ax.set_xlabel("u")
    ax.set_ylabel("FD Score")
    fig.savefig("./%i_Accuracy_classification_AB.pdf"%k)

def Classification_error_AB_EP_20_v_39():  
   #Plotting 61% vs 74% of 31,39 attributes
    L2=[0.6158,0.6679,0.454,0.4795,0.7537,0.8926,0.7459,0.7916]    
    L1=[0.6038,0.6613,0.4344,0.5368,0.7403,0.8911,0.7383,0.7859]
    IS=[0.5869,0.6477,0.4149,0.3657,0.7208,0.8844,0.7238,0.7721]
    SP=[0.5806,0.6426,0.4075,0.3015,0.7135,0.8819,0.7184,0.7669]
    WD=[0.6038,0.6613,0.4344,0.5368,0.7403,0.8911,0.7383,0.7859]
    
    barWidth = 1/7
    br1 = np.arange(len(L2))
    br2 = [x+0.025 + barWidth for x in br1]
    br3 = [x+0.025 + barWidth for x in br2]
    br4 = [x+0.025 + barWidth for x in br3]
    br5 = [x+0.025 + barWidth for x in br4]
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    
    
    
    ax.bar(br1, L2, color =palette['L2'], width = barWidth,
            edgecolor ='grey', label ='L2')
    ax.bar(br2, L1, color =palette['L1'], width = barWidth,
            edgecolor ='grey', label ='L1')
    ax.bar(br3, IS, color =palette['IS'], width = barWidth,
            edgecolor ='grey', label ='IS')
    ax.bar(br4, SP, color =palette['SP'], width = barWidth,
            edgecolor ='grey', label ='SP')
    ax.bar(br5, WD, color =palette['WD'], width = barWidth,
            edgecolor ='grey', label ='WD')
    
    #Labels
    # ax.set_xticklabels([r + barWidth for r in range(len(L2))],
    #         ["[0,1]", "[1,0]", "[0,1]", "[1,0]", "[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]","[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]"])

    ax.set_xticks([x+0.3 for x in range(8)])
    ax.set_xticklabels(["[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]","[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]"])
    ax.legend()
    ax2=ax.twiny()
    ax2.set_xticks([0.3,0.7])
    ax2.set_xticklabels(["Young,black hair $Acc_{avg}=65\%$","gender,black hair $Acc_{avg}=84\%$"])
    ax.legend( prop={'size': 18})
    
    ax.set_xlabel("u")
    ax.set_ylabel("FD Score")
    fig.savefig("./%i_Accuracy_classification_AB_20_39.pdf"%4)
    
def Classification_error_AB_EP_same_At():  
   #Plotting 61% vs 74% of 31,39 attributes
    L2=[0.3538,0.6479,0.4856,0.6970,0.7364,0.6824,0.5576,0.6702]    
    L1=[0.3912,0.6420,0.5552,0.6884,0.7331,0.6726,0.5233,0.6562]
    IS=[0.2725,0.6298,0.3423,0.6726,0.7231,0.6556,0.4954,0.6385]
    SP=[0.2280,0.6253,0.2624,0.6667,0.7194,0.6493,0.4849,0.6318]
    WD=[0.3912,0.6420,0.5552,0.6884,0.7331,0.6726,0.5233,0.6562]
    
    barWidth = 1/7
    br1 = np.arange(len(L2))
    br2 = [x+0.025 + barWidth for x in br1]
    br3 = [x+0.025 + barWidth for x in br2]
    br4 = [x+0.025 + barWidth for x in br3]
    br5 = [x+0.025 + barWidth for x in br4]
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    
    
    
    ax.bar(br1, L2, color =palette['L2'], width = barWidth,
            edgecolor ='grey', label ='L2')
    ax.bar(br2, L1, color =palette['L1'], width = barWidth,
            edgecolor ='grey', label ='L1')
    ax.bar(br3, IS, color =palette['IS'], width = barWidth,
            edgecolor ='grey', label ='IS')
    ax.bar(br4, SP, color =palette['SP'], width = barWidth,
            edgecolor ='grey', label ='SP')
    ax.bar(br5, WD, color =palette['WD'], width = barWidth,
            edgecolor ='grey', label ='WD')
    
    #Labels
    # ax.set_xticklabels([r + barWidth for r in range(len(L2))],
    #         ["[0,1]", "[1,0]", "[0,1]", "[1,0]", "[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]","[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]"])

    ax.set_xticks([x+0.3 for x in range(8)])
    ax.set_xticklabels(["[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]","[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]"])
    ax.legend()
    ax2=ax.twiny()
    ax2.set_xticks([0.3,0.7])
    ax2.set_xticklabels(["Young,Smiling $Acc_{avg}=61\%$","Young,Smiling $Acc_{avg}=74\%$"])
    ax.legend( prop={'size': 18})
    
    ax.set_xlabel("u")
    ax.set_ylabel("FD Score")
    fig.savefig("./%i_Accuracy_classification_AB_31_39.pdf"%4)

def Classification_error_Fair_EP(k=2):  
    if k==2:
        L2=[0.0153,0.1015]
        L1=[0.0153,0.1015]
        IS=[0.0153,0.1015]
        SP=[0.0153,0.1015]
        WD=[0.0153,0.1015]
    else:     
        L2=[0.0583,0.0751]
        L1=[0.0544,0.0644]
        IS=[0.0515,0.053]
        SP=[0.0504,0.0487]
        WD=[0.0544,0.0644]

        
    barWidth = 1/7
    br1 = np.arange(len(L2))
    br2 = [x+0.025 + barWidth for x in br1]
    br3 = [x+0.025 + barWidth for x in br2]
    br4 = [x+0.025 + barWidth for x in br3]
    br5 = [x+0.025 + barWidth for x in br4]
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    
    
    
    ax.bar(br1, L2, color =palette['L2'], width = barWidth,
            edgecolor ='grey', label ='L2')
    ax.bar(br2, L1, color =palette['L1'], width = barWidth,
            edgecolor ='grey', label ='L1')
    ax.bar(br3, IS, color =palette['IS'], width = barWidth,
            edgecolor ='grey', label ='IS')
    ax.bar(br4, SP, color =palette['SP'], width = barWidth,
            edgecolor ='grey', label ='SP')
    ax.bar(br5, WD, color =palette['WD'], width = barWidth,
            edgecolor ='grey', label ='WD')
    
    #Labels
    # ax.set_xticklabels([r + barWidth for r in range(len(L2))],
    #         ["[0,1]", "[1,0]", "[0,1]", "[1,0]", "[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]","[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]"])
    if k==2:
        ax.set_xticks([x+0.3 for x in range(2)])
        ax.set_xticklabels(["[0.5,0.5]","[0.5,0.5]"])
        ax.legend()
        ax2=ax.twiny()
        ax2.set_xticks([0.2,0.8])
        ax2.set_xticklabels(["Male-Female","Young-Old"])
    else:
        ax.set_xticks([x+0.3 for x in range(2)])
        ax.set_xticklabels(["[0.5,0.5]","[0.5,0.5]"])
        ax.legend()
        ax2=ax.twiny()
        ax2.set_xticks([0.2,0.8])
        ax2.set_xticklabels(["Male-Female,Black Hair","Young-Old,Smiling"])
    ax.legend( prop={'size': 18})
    plt.ylim(0, 0.2)
    
    ax.set_xlabel("u")
    ax.set_ylabel("FD Score")
    fig.savefig("./%i_Accuracy_classification_Fair.pdf"%k)

def Classification_error_Fair_EP_same_At():  
    
    L2=[0.1585,0.0751]
    L1=[0.1740,0.0644]
    IS=[0.1240,0.053]
    SP=[0.1054,0.0487]
    WD=[0.1740,0.0644]

        
    barWidth = 1/7
    br1 = np.arange(len(L2))
    br2 = [x+0.025 + barWidth for x in br1]
    br3 = [x+0.025 + barWidth for x in br2]
    br4 = [x+0.025 + barWidth for x in br3]
    br5 = [x+0.025 + barWidth for x in br4]
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    
    
    
    ax.bar(br1, L2, color =palette['L2'], width = barWidth,
            edgecolor ='grey', label ='L2')
    ax.bar(br2, L1, color =palette['L1'], width = barWidth,
            edgecolor ='grey', label ='L1')
    ax.bar(br3, IS, color =palette['IS'], width = barWidth,
            edgecolor ='grey', label ='IS')
    ax.bar(br4, SP, color =palette['SP'], width = barWidth,
            edgecolor ='grey', label ='SP')
    ax.bar(br5, WD, color =palette['WD'], width = barWidth,
            edgecolor ='grey', label ='WD')
    
    #Labels
    # ax.set_xticklabels([r + barWidth for r in range(len(L2))],
    #         ["[0,1]", "[1,0]", "[0,1]", "[1,0]", "[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]","[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]"])

    ax.set_xticks([x+0.3 for x in range(2)])
    ax.set_xticklabels(["[0.5,0.5]","[0.5,0.5]"])
    ax.legend()
    ax2=ax.twiny()
    ax2.set_xticks([0.2,0.8])
    ax2.set_xticklabels(["Young,Smiling $Acc_{avg}=61%$","Young,Smiling $Acc_{avg}=74%$"])
    ax.legend( prop={'size': 18})
    plt.ylim(0, 0.2)
    
    ax.set_xlabel("u")
    ax.set_ylabel("FD Score")
    fig.savefig("./%i_Accuracy_classification_Fair_31_39.pdf"%2)
    
def Classification_error_Fair_EP_20_v_39():  
    
    L2=[0.1347,0.0153]
    L1=[0.1412,0.0153]
    IS=[0.1145,0.0153]
    SP=[0.1047,0.0153]
    WD=[0.1412,0.0153]

        
    barWidth = 1/7
    br1 = np.arange(len(L2))
    br2 = [x+0.025 + barWidth for x in br1]
    br3 = [x+0.025 + barWidth for x in br2]
    br4 = [x+0.025 + barWidth for x in br3]
    br5 = [x+0.025 + barWidth for x in br4]
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    
    
    
    ax.bar(br1, L2, color =palette['L2'], width = barWidth,
            edgecolor ='grey', label ='L2')
    ax.bar(br2, L1, color =palette['L1'], width = barWidth,
            edgecolor ='grey', label ='L1')
    ax.bar(br3, IS, color =palette['IS'], width = barWidth,
            edgecolor ='grey', label ='IS')
    ax.bar(br4, SP, color =palette['SP'], width = barWidth,
            edgecolor ='grey', label ='SP')
    ax.bar(br5, WD, color =palette['WD'], width = barWidth,
            edgecolor ='grey', label ='WD')
    
    #Labels
    # ax.set_xticklabels([r + barWidth for r in range(len(L2))],
    #         ["[0,1]", "[1,0]", "[0,1]", "[1,0]", "[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]","[1,0,0,0]","[0,1,0,0]","[0,0,1,0]","[0,0,0,1]"])

    ax.set_xticks([x+0.3 for x in range(2)])
    ax.set_xticklabels(["[0.5,0.5]","[0.5,0.5]"])
    ax.legend()
    ax2=ax.twiny()
    ax2.set_xticks([0.2,0.8])
    ax2.set_xticklabels(["Young,black hair $Acc_{avg}=65\%$","gender,black hair $Acc_{avg}=84\%$"])
    ax.legend( prop={'size': 18})
    plt.ylim(0, 0.2)
    
    ax.set_xlabel("u")
    ax.set_ylabel("FD Score")
    fig.savefig("./%i_Accuracy_classification_Fair_20_39.pdf"%4)

'''
Sweep Plot
'''
attr=8
index=1
def plot_sweep(attr,index):
    styles=['_','--','-.',',',':','-']
    label_list=["Count","L2","L1","IS","Specificity","WD","Linear"]
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    
    def pre_process(attr,index):
        path=("../Sweep_{}_multi/log_stamford_fair_norm_raw_{}.txt".format(attr,index))
        data=pd.read_csv(path)
        count=data.iloc[:,1].to_numpy()
        L2=data.iloc[:,2].to_numpy()
        L1=data.iloc[:,3].to_numpy()
        IS=data.iloc[:,4].to_numpy()
        SP=data.iloc[:,5].to_numpy()
        WD=data.iloc[:,6].to_numpy()
        linear=[(len(L2)-(i+1))/len(L2) for i in range(len(L2))]
        return(count,L2,L1,IS,SP,WD,linear)

    count,L2,L1,IS,SP,WD,linear=pre_process(attr,index)
    array_list=[count,L2,L1,IS,SP,WD,linear]

    for i in range(1,len(label_list)):
        ax.plot(array_list[0], array_list[i],styles[i-1],label=label_list[i])
        ax.legend( prop={'size': 20})
        plt.plot(array_list[0],array_list[i],styles[i-1])
    plt.show()

#Plot Metrics upper bounds
def plot_upperbound(metric,norm=0):
    colours=["0.6","0.4","0.2","0.0"]
    # plot_sweep(8,1)
    #Plot instantiate
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    marker=["o","x","s","v"]
    at=[2,4,8,16]
    labels=["k=2","k=4","k=8","k=16"]
    for i in range(len(at)):
        if norm==0:
            data_path_ideal="./csv/{}_report_at_{}.csv".format(metric,at[i])
        else:
            data_path_ideal="./csv/{}_report_at_{}_normalised.csv".format(metric,at[i])
        data_raw=pd.read_csv(data_path_ideal)
        delta_p=data_raw.iloc[:,0].to_numpy()
        delta_f=data_raw.iloc[:,3].to_numpy()
        instances=np.linspace(1,len(delta_p),len(delta_p))
        
        ax.plot(instances,delta_f-delta_p,marker[i],label=labels[i],c=colours[i],alpha=0.5,markersize=12)
    ax.legend( prop={'size': 24})
    ax.set_xlabel("Random Sampling Instance")
    ax.set_ylabel("$|\Delta f|$ - $|\Delta p|$")
    if norm==0:
        fig.savefig("./{}_error_unnormalised.pdf".format(metric))
    else:
        fig.savefig("./{}_error_normalised.pdf".format(metric))
        
#Plotting the error accuracy diagrams
def error_plot_demo():
    sns.color_palette("rocket", as_cmap=True)
    plt.rcParams.update({'font.size': 16})
    # loading dataset
    data = pd.read_csv("./error_demo/data.csv")
      
    # draw pointplot
    fig=sns.catplot(x = "Bias Percentage",
                  y = "Score",
                  hue="Attributes",
                  col= "Metric",
                  data = data,
                  kind="point",
                  dodge=True,
                  height=4, 
                  aspect=.7,
                  palette=['black', 'grey']);
    # # show the plot
    plt.show()
    fig.savefig("./error_ub_demo.pdf")
    
# array=["L1","L2","Sp","Is"]
# for i in array:
#     plot_upperbound(i,0)

# for i in range(2,6,2):
#     Classification_error_AB_EP(i)
#     Classification_error_Fair_EP(i)

error_plot_demo()