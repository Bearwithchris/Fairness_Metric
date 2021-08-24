# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:05:18 2021

@author: Chris
"""
import numpy as np
import copy
import utils
from tqdm import tqdm


#Init parameters 
k=4 #Cardinality of attributes (constant)
ibias=0.7 #initialise bias (constant)
cAcc=0.5 #classifier's accuracy (independent variable)
lSamples=1000 #Number of label samples (constant)
pSamples=100 #Number of samples for calculating p (constant)

pred=np.zeros([pSamples,k])
fScoresL2=np.zeros(pSamples)
fScoresL1=np.zeros(pSamples)
fScoresIs=np.zeros(pSamples)
fScoresSp=np.zeros(pSamples)
#Generate label grould truth with ibias
Lgt=utils.populate_Lgt(ibias,k,lSamples) 
for i in tqdm(range(pSamples)):
    #Generate Lpred
    Lpred=utils.gen_pred_labels(k,ibias,cAcc,Lgt,lSamples,bias_flip=0)
    #Calculate the p of the Lpred
    p=utils.L2P(Lpred)
    #Calculate the f
    f=utils.fairness_discrepancy(p, k, norm=1)
    
    #Log the scores
    pred[i,:]=p
    fScoresL2[i]=f[0]
    fScoresL1[i]=f[1]
    fScoresIs[i]=f[2]
    fScoresSp[i]=f[3]    

np.savez("p_bias_{}_Acc_{}_k_{}".format(ibias,cAcc,k),x=pred)
np.savez("bias_{}_Acc_{}_fScores_k_{}".format(ibias,cAcc,k),L2=fScoresL2,L1=fScoresL1,Is=fScoresIs,Sp=fScoresSp)
    




