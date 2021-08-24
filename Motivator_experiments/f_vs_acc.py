# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:05:18 2021

@author: Chris
"""
import numpy as np
import copy
import utils
from tqdm import tqdm
import pandas as pd


#Init parameters 
k=2 #Cardinality of attributes (constant)
ibias=0.9 #initialise bias (constant)

# k=2
u_acc_array_k2=[[0.9,0.9],[0.8,0.8],[0.7,0.7],[0.6,0.6],[0.5,0.5]]
u_acc_array_k2_sd1=[[1.0,0.8],[0.9,0.7],[0.8,0.6]]


#k=4
u_acc_array_sd0=[[0.9,0.9,0.9,0.9],[0.8,0.8,0.8,0.8],[0.7,0.7,0.7,0.7]] #classifier's accuracy (independent variable)
u_acc_array_sd1=[[0.8,1.0,1.0,0.8],[0.7,0.9,0.9,0.7],[0.6,0.8,0.8,0.6],[0.5,0.7,0.7,0.5]] #classifier's accuracy (independent variable)
u_acc_array_sd2=[[0.6,1.0,1.0,0.6],[0.5,0.9,0.9,0.5],[0.4,0.8,0.8,0.4]] #classifier's accuracy (independent variable)


lSamples=500 #Number of label samples (constant)
pSamples=100 #Number of samples for calculating p (constant)

u_acc=u_acc_array_k2[0]

pred=np.zeros([pSamples,k])
fScoresL2=np.zeros(pSamples)
fScoresL1=np.zeros(pSamples)
fScoresIs=np.zeros(pSamples)
fScoresSp=np.zeros(pSamples)
#Generate label grould truth with ibias
Lgt=utils.populate_Lgt(ibias,k,lSamples) 
for i in tqdm(range(pSamples)):
    #Generate Lpred
    Lpred=utils.gen_pred_labels(k,ibias,Lgt,lSamples,u_acc,bias_flip=0)
    #Calculate the p of the Lpred
    p=utils.L2P(Lpred)
    #Calculate the f
    f=utils.fairness_discrepancy(p, k, norm=0)
    
    #Log the scores
    pred[i,:]=p
    fScoresL2[i]=f[0]
    fScoresL1[i]=f[1]
    fScoresIs[i]=f[2]
    fScoresSp[i]=f[3]    

#Combined accuracy i.e., Acc_avg
cAcc=np.around(sum(u_acc)/len(u_acc),2)
#Save p
np.savez("lsample_{}_p_bias_{}_Acc_{}_k_{}".format(lSamples,ibias,cAcc,k),x=pred)
#Save f
# np.savez("bias_{}_Acc_{}_fScores_k_{}".format(ibias,cAcc,k),L2=fScoresL2,L1=fScoresL1,Is=fScoresIs,Sp=fScoresSp)

d={'L2':fScoresL2,'L1':fScoresL1,'Is':fScoresIs,'Sp':fScoresSp}
df=pd.DataFrame(data=d)
df.to_csv("Compiled_score_bias_{}_Acc_{}_k_{}.csv".format(ibias,cAcc,k))




