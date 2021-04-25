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
parser.add_argument('--multi_class_idx',nargs="*", type=int, help='CelebA class label for training.', default=[6,7,8])
parser.add_argument('--multi', type=bool, default=True, help='If True, runs multi-attribute classifier')
parser.add_argument('--split_type', type=str, help='[train,val,split]', default="test")
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

# def test_gen_ref(perc_f=0.5,sample_size=10000):
#     """
#     Returns a dataset used for classification for given class label <class_idx>. If class_idx2 is not None, returns both labels (this is typically used for downstream tasks)
    
#     Args:
#         split (str): one of [train, val, test]
#         class_idx (int): class label for protected attribute
#         class_idx2 (None, optional): additional class for downstream tasks
    
#     Returns:
#         TensorDataset for training attribute classifier
#     """
#     # class_idx=20
#     split=args.split_type
#     if not args.multi:
#         data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format(split))
#         labels = torch.load(BASE_PATH + '{}_labels_celeba_64x64.pt'.format(split))
#         labels = labels[:, class_idx]
#     else:
#         data = torch.load(BASE_PATH + '{}_multi_even_celeba_64x64.pt'.format(split))
#         labels = torch.load(BASE_PATH + '{}_multi_even_celeba_64x64.pt'.format(split))
        
    
#     #Seiving out the male_female breakdown
#     M_F_labels=labels.numpy()[:,class_idx]
#     male=np.where(M_F_labels==1)
#     female=np.where(M_F_labels==0)
    
#     #Split counts
#     #total_Count=len(male[0])+len(female[0])
#     m_Count=round(sample_size*(1-perc_f))
#     f_Count=round(sample_size*perc_f)
    
#     #Extracting data
#     data_M=np.take(data,male,axis=0)[0]
#     data_F=np.take(data,female,axis=0)[0]
    
#     #Random selection index
#     sample_index_M=np.random.choice(len(data_M), int(m_Count))
#     sample_index_F=np.random.choice(len(data_F), int(f_Count))
#     data_M=np.take(data_M,sample_index_M,axis=0)
#     data_F=np.take(data_F,sample_index_F,axis=0)
#     rebalanced_dataset=torch.cat((data_M,data_F),axis=0)
    
#     #Newlabels
#     label_M=torch.ones(len(data_M))
#     label_F=torch.zeros(len(data_F))
#     rebalanced_labels=torch.cat((label_M,label_F))
#     # torch.save((rebalanced_dataset,rebalanced_labels),'./test_fairness_data_%f_%f_samples_%i'%(round(perc_f,2),round(1-perc_f,2),sample_size))

#     if not (os.path.exists('./real_data')):
#         os.makedirs('./real_data')
        
#     torch.save((rebalanced_dataset,rebalanced_labels),'./real_data/test_fairness_ref_data_%f_%f_samples_%i'%(round(perc_f,2),round(1-perc_f,2),sample_size))

#     # return appropriate split

#     dataset = torch.utils.data.TensorDataset(rebalanced_dataset, rebalanced_labels)
#     return dataset
def sample_max(dist):
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
    # #Seiving out the male_female breakdown
    # M_F_labels=labels.numpy()[:,class_idx]
    # male=np.where(M_F_labels==1)
    # female=np.where(M_F_labels==0)
    
    # #Split counts
    # #total_Count=len(male[0])+len(female[0])
    # m_Count=round(sample_size*(1-perc_f))
    # f_Count=round(sample_size*perc_f)
    
    # #Extracting data
    # data_M=np.take(data,male,axis=0)[0]
    # data_F=np.take(data,female,axis=0)[0]
    
    # #Random selection index
    # sample_index_M=np.random.choice(len(data_M), int(m_Count))
    # sample_index_F=np.random.choice(len(data_F), int(f_Count))
    # data_M=np.take(data_M,sample_index_M,axis=0)
    # data_F=np.take(data_F,sample_index_F,axis=0)
    # rebalanced_dataset=torch.cat((data_M,data_F),axis=0)
    
    # #Newlabels
    # label_M=torch.ones(len(data_M))
    # label_F=torch.zeros(len(data_F))
    # rebalanced_labels=torch.cat((label_M,label_F))
    # # torch.save((rebalanced_dataset,rebalanced_labels),'./test_fairness_data_%f_%f_samples_%i'%(round(perc_f,2),round(1-perc_f,2),sample_size))

    # if not (os.path.exists('./real_data')):
    #     os.makedirs('./real_data')
        
    # torch.save((rebalanced_dataset,rebalanced_labels),'./real_data/test_fairness_data_%f_%f_samples_%i'%(round(perc_f,2),round(1-perc_f,2),sample_size))

    # # return appropriate split

    # dataset = torch.utils.data.TensorDataset(rebalanced_dataset, rebalanced_labels)
    # return dataset

#Multu===================================================================================

if __name__=='__main__':
    testdist=dist(2**len(args.multi_class_idx))
    cap=sample_max(testdist[0])
    f=open("../logs/data_tags.txt","a")
    for i in range(len(testdist)):
        generate_test_datasets(testdist[i],i,cap)
    # # Standard test with regards to percentages
    # unbias_perc=0.5
    # samples=10000
    # test_gen_ref(unbias_perc,samples)
    # # for i in np.arange (0.1,1,0.1):
    # #     test_gen(i,samples) #0.5 Female
    # test_gen(unbias_perc,samples) #0.5 Female
        
    #Standard test with regards to Samples
    # unbias_perc=0.5
    # bias_perc=0.7
    # for i in np.arange (1000,21000,1000):
    #     test_gen_ref(unbias_perc,i)
    #     test_gen(bias_perc,i) #0.5 Female
    # for i in np.arange (30000,81000,10000):
    #     test_gen_ref(unbias_perc,i)
    #     test_gen(bias_perc,i) #0.5 Female
