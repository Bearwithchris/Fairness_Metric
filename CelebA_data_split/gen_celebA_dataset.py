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
args = parser.parse_args()


def dist(count):
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
                dist_base[i]=dist_base[i]+0.5
                dist_base[count-1-i]=dist_base[count-1-i]-0.5
            array=copy.deepcopy(dist_base)
            array=array/np.sum(array)
            dist_array.append(array)
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
    cap=minCount/max(dist)
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
        label_arg[point:point+dist_count[i]]=np.random.choice(np.where(labels==i)[0],dist_count[i],replace=False)
        point=point+dist_count[i]
        
    new_data= data[label_arg,:,:,:] #Even data
    new_labels=labels[label_arg]
    new_tag="attr_"+str(attributes)+"_"+str(dist).strip("[").strip("]").replace(" ","_")
    f.write("gen_data_%s_%s\n"%(index,new_tag))
    torch.save((new_data,new_labels),'../data/resampled_ratio/gen_data_%i_%s'%(attributes,index))
    return new_labels


#Multu===================================================================================

if __name__=='__main__':
    if args.mode_normal==1:
        testdist=dist(2**len(args.multi_class_idx))
    else:
        testdist=extreme_dist(2**len(args.multi_class_idx))
    
    print(args.mode_normal)
    cap=sample_max(testdist[0])
    f=open("../logs/data_tags.txt","a")
    for i in range(len(testdist)):
        new_labels=generate_test_datasets(testdist[i],i,cap)

