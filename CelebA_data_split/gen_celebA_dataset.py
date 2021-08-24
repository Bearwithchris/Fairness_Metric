# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:27:51 2021

@author: Chris
"""
import torch 
import numpy as np
import os
import argparse
import copy

BASE_PATH = '../data/'
parser = argparse.ArgumentParser()
parser.add_argument('--class_idx', type=int, help='CelebA class label for training.', default=20)
parser.add_argument('--multi_class_idx',nargs="*", type=int, help='CelebA class label for training.', default=[39])
parser.add_argument('--multi', type=int, default=1, help='If True, runs multi-attribute classifier')
parser.add_argument('--split_type', type=str, help='[train,val,split]', default="test")
parser.add_argument('--mode_normal', type=int, default=1, help='If True, normal mode, else extreme mode')
parser.add_argument('--ABEP', type=int, default=0, help='State the Absolute bias starting point index')
parser.add_argument('--step_mul', type=int, default=1, help='defines the dist step size')

#For bias one experiment
parser.add_argument('--bias_one_perc', type=float, default=0.9, help='defines the size of the bias one dist, only if mode_normal==4')
args = parser.parse_args()

def dist(count):
    step=1/2**args.step_mul
# def dist(count,step_mul=1):
#     step=1/2**step_mul
    dist_base=np.linspace(1,count,count)
    even=np.sum(dist_base)/count
    target=np.ones(count)*even
    
    steps=int(count/2)
    dist_array=[]
    
    #Intial Base dist
    array=copy.deepcopy(dist_base)
    array=array/np.sum(array)
    dist_array.append(array)
    while (np.array_equal(dist_base,target)!=True):
        for i in range(steps):
            if ((dist_base[i]==even) and ((dist_base[count-1-i]==even))):
                break
            else:
                dist_base[i]=dist_base[i]+step
                dist_base[count-1-i]=dist_base[count-1-i]-step
            array=copy.deepcopy(dist_base)
            array=array/np.sum(array)
            dist_array.append(array)
    return dist_array

#Create variable starting points for the bias distribution=1 (FIXED)
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

#Extension of dist 2 but variable starting points for the bias distribution
def dist3(count,start_index=1):
    if count==2 or count==4:
        step=1
    elif count==8:
        step=0.5
    else: #count==16
        step=0.25
    #Make perfectly bias
    dist_base=np.zeros(count)
    dist_base[start_index]=100
    
    #Target
    even=100/count
    target=np.ones(count)*even
    
    #Loop parameters
    dist_array=[]
    
    #Allocate the starting point of the populating index
    if start_index==0:
        index=1
    else:
        index=0
        
    while (np.array_equal(dist_base,target)!=True):
        if (dist_base[index]!=target[index]):
            #Transfer from index 0
            dist_base[start_index]=dist_base[start_index]-step
            dist_base[index]=dist_base[index]+step
        else:
            index+=1
            if index==start_index:
                index+=1
        array=copy.deepcopy(dist_base) 
        array=array/100
        dist_array.append(array)
        
    return dist_array

    
#Random distribution 
def random_dist(attributeS,samples=1000):
    dist_array=[]
    for i in range(samples):
        rng = np.random.default_rng()
        random_dist = rng.random((1, attributeS))
        norm_random_dist=random_dist/random_dist.sum()
        dist_array.append(norm_random_dist[0])
    
    return dist_array

def extreme_dist(count):
    dist_array=[]
    for i in range(count):
        bias=np.zeros(count)
        bias[i]=1
        dist_array.append(bias)
    unbias=np.ones(count)*(1./count)
    dist_array.append(unbias)
    return dist_array

#Generate xx_.. distribution
def dist_bias_one(bias=0.9,at_size=4,samples=1000):
    dist_array=[]
    for _i in range(samples):
        array_temp=np.zeros(at_size)
        order=np.arange(at_size)
        np.random.shuffle(order)
        for j in range(len(order)):
            if j==0:
                array_temp[order[j]]=bias
            else:
                array_temp[order[j]]=(1-bias)/(at_size-1)
        dist_array.append(np.array(array_temp))
    return dist_array
            
         
            

def sample_max(dist):
    class_idx=args.class_idx
    split=args.split_type
    if args.multi==0:
        data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format(split))
        labels = torch.load(BASE_PATH + '{}_labels_celeba_64x64.pt'.format(split))
        labels = labels[:, class_idx]
        attributes=2
        class_count=2
    else:
        data = torch.load(BASE_PATH + '{}_multi_even_data_celeba_64x64.pt'.format(split))
        labels = torch.load(BASE_PATH + '{}_multi_even_labels_celeba_64x64.pt'.format(split))
        attributes=2**(len(args.multi_class_idx))
        class_count=len(args.multi_class_idx)
        
    #Determine the number of samples per class (even)
    minCount=162770
    for i in range((attributes)):
        count=len(np.where(labels==i)[0])
        if count<minCount:
            minCount=count
    # cap=minCount/max(dist)
    cap=minCount
    return cap


def generate_test_datasets(dist,index,cap):
    """
    Returns a dataset used for classification for given class label <class_idx>. If class_idx2 is not None, returns both labels (this is typically used for downstream tasks)
    
    Args:
        split (str): one of [train, val, test]
        class_idx (int): class label for protected attribute
        class_idx2 (None, optional): additional class for downstream tasks
    
    Returns:
        TensorDataset for training attribute classifier
    """
    #Retrieve database
    class_idx=args.class_idx
    split=args.split_type
    if not args.multi:
        data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format(split))
        labels = torch.load(BASE_PATH + '{}_labels_celeba_64x64.pt'.format(split))
        labels = labels[:, class_idx]
        attributes=2
        class_count=2
    else:
        data = torch.load(BASE_PATH + '{}_multi_even_data_celeba_64x64.pt'.format(split))
        labels = torch.load(BASE_PATH + '{}_multi_even_labels_celeba_64x64.pt'.format(split))
        attributes=2**(len(args.multi_class_idx))
        class_count=len(args.multi_class_idx)
         

    #Determine the number of samples per class (even)
    # minCount=162770
    # for i in range((attributes)):
    #     count=len(np.where(labels==i)[0])
    #     if count<minCount:
    #         minCount=count
            
    # label_arg=np.ones(minCount*attributes)

    dist_count=np.round((cap*dist)).astype(int)
    label_arg=np.ones(np.sum(dist_count))
    point=0
    for i in range(attributes):
        try:
            label_arg[point:point+dist_count[i]]=np.random.choice(np.where(labels==i)[0],dist_count[i],replace=False)
        except:
            print("error")
        point=point+dist_count[i]
        
    new_data= data[label_arg,:,:,:] #Even data
    new_labels=labels[label_arg]
    new_tag="attr_"+str(attributes)+"_"+str(dist).strip("[").strip("]").replace(" ","_")
    f.write("gen_data_%s_%s\n"%(index,new_tag))
    torch.save((new_data,new_labels),'../data/resampled_ratio/gen_data_%i_%s'%(attributes,index))
    return new_labels


#Multu===================================================================================

if __name__=='__main__':
    #"Normal" distributions method
    if args.mode_normal==1:
        testdist=dist(2**len(args.multi_class_idx))
    #shifting distribution method @ different AB-EP
    elif args.mode_normal==2:
        testdist=dist3(2**len(args.multi_class_idx),args.ABEP)
        print (args.ABEP)
    #Random distirbution methods
    elif args.mode_normal==3:
        testdist=random_dist(2**len(args.multi_class_idx))
    #xx-... distribution (bias one and even dist on the rest)
    elif args.mode_normal==4:
        testdist=dist_bias_one(args.bias_one_perc,2**len(args.multi_class_idx))
    #Only extreme points
    else:
        testdist=extreme_dist(2**len(args.multi_class_idx))
    
    #Save ideal dist for testing later
    np.savez("../logs/ideal_dist.npz",x=testdist)
    
    # print(args.mode_normal)
    cap=sample_max(testdist[0])
    f=open("../logs/data_tags.txt","a")
    for i in range(len(testdist)):
        new_labels=generate_test_datasets(testdist[i],i,cap)

