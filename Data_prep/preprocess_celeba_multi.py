# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:40:59 2021

@author: Chris
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from IPython.display import Image
import torch
import argparse


DATA_DIR = '../data/'


parser = argparse.ArgumentParser()
parser.add_argument('--split_type', type=str, help='[train,val,split]', default="train")
parser.add_argument('--class_idx',nargs="*", type=int, help='CelebA class label for training.', default=[6,7,8,20])
args = parser.parse_args()
# args.cuda = args.cuda and torch.cuda.is_available()

attr_lookup=np.array(["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald","Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair","Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"
])
attr_name=[]
for attr in args.class_idx:
    attr_name.append(attr_lookup[attr])

print(attr_name)


# make sure to have pre-processed the celebA dataset before running this code!
  # repeat for train and val
data = torch.load(os.path.join(DATA_DIR, '{}_celeba_64x64.pt'.format(args.split_type)))
labels = torch.load(os.path.join(DATA_DIR, '{}_labels_celeba_64x64.pt'.format(args.split_type)))
new_labels = np.zeros(len(labels))
unique_items = np.unique(labels[:,args.class_idx], axis=0)
for i, unique in enumerate(unique_items):
    yes = np.ravel([np.array_equal(x,unique) for x in labels[:,args.class_idx]])
    new_labels[yes] = i
    count=len(yes[yes==True])
    print(unique, i, "count=%i"%count)
    

new_labels = torch.from_numpy(new_labels)
torch.save(new_labels, os.path.join(DATA_DIR, '{}_multi_labels_celeba_64x64.pt'.format(args.split_type)))

