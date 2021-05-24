''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import time
import os
import glob
import sys

import functools
import math
import numpy as np
from tqdm import tqdm, trange
import pickle


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
# import inception_utils
# import utils
# import utils_add_on as uao
# import losses
# from clf_models import ResNet18, BasicBlock, Net
# import fid_score_mod

import fid_score_mod_AE
import argparse


def load_data(attributes,index):
    data=torch.load('../data/resampled_ratio/gen_data_ref_%i_%s'%(attributes,index))
    print ("Data loaded: "+'../data/resampled_ratio/gen_data_ref_%i_%s'%(attributes,index))
    dataset=data[0]
    labels=data[1]
    train_set = torch.utils.data.TensorDataset(dataset)
    return (train_set,len(data[0]),data[1])


def classify_examples(model, sample_path):
    """
    classifies generated samples into appropriate classes 
    """
    model.eval()
    preds = []
    probs = []
    samples = np.load(sample_path)['x']
    n_batches = samples.shape[0] // 1000
    print (sample_path)

    with torch.no_grad():
        # generate 10K samples
        for i in range(n_batches):
            x = samples[i*1000:(i+1)*1000]
            samp = x / 255.  # renormalize to feed into classifier
            samp = torch.from_numpy(samp).to('cuda').float()

            # get classifier predictions
            logits, probas = model(samp)
            _, pred = torch.max(probas, 1) #Returns the max indices i.e. index
            probs.append(probas)
            preds.append(pred)
        preds = torch.cat(preds).data.cpu().numpy()
        probs = torch.cat(probs).data.cpu().numpy()
        # probs = torch.cat(probs).data.cpu()

    return preds, probs

def run():
    # Prepare state dict, which holds things like epoch # and itr #
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_path', type=str, help='Folder of the CLF file', default="attr_clf")
    # parser.add_argument('--multi_clf_path', type=str, help='Folder of the Multi CLF file', default="multi_clf")
    parser.add_argument('--index', type=int, help='dataset index to load', default=0)
    parser.add_argument('--class_idx', type=int, help='CelebA class label for training.', default=20)
    # parser.add_argument('--multi_class_idx',nargs="*", type=int, help='CelebA class label for training.', default=[6,7,8,20])
    parser.add_argument('--multi_class_idx',nargs="*", type=int, help='CelebA class label for training.', default=[20]) #default=[39,31])
    parser.add_argument('--multi', type=int, default=1, help='If True, runs multi-attribute classifier')
    parser.add_argument('--split_type', type=str, help='[train,val,split]', default="test")
    args = parser.parse_args()


    # CLF_PATH = '../Data_prep/results/%s/model_best.pth.tar'%args.clf_path
    # MULTI_CLF_PATH = '../Data_prep/results/%s/model_best.pth.tar'%args.multi_clf_path
    device = 'cuda'

    torch.backends.cudnn.benchmark = True


    #Log Runs
    # f=open('../%s/log_stamford_fair.txt' %("logs"),"a")
    # fnorm=open('../%s/log_stamford_fair_norm.txt' %("logs"),"a")

    # experiment_name = (config['experiment_name'] if config['experiment_name'] #Default CelebA
    #                     else utils.name_from_config(config))
    
    #Load dataset to be tested
    if args.multi==1:
        attributes=2**len(args.multi_class_idx)
    else: 
        attributes=2 #Single class
    train_set,size,labels=load_data(attributes, args.index)
    
    # # classify examples and get probabilties
    # n_classes = 2
    # if config['multi']:
    #     n_classes = 4
   
    print ("Preparing data....")
    print ("Dataset has a total of %i data instances"%size)
    k=0
    
    file="../data/FID_sample_storage_%i"%attributes
    if (os.path.exists(file)!=True):
        os.makedirs(file)
    npz_filename = '%s/%s_fid_real_samples_ref_%s.npz' % (file,attributes, args.index) #E.g. perc_fid_samples_0
    if os.path.exists(npz_filename):
        print('samples already exist, skipping...')
    else:
        X = []
        pbar = tqdm(train_set)
        print('Sampling images and saving them to npz...' ) #10k
        count=1 
        
        for i ,x in enumerate(pbar):
            X+=x                
        X=np.array(torch.stack(X)).astype(np.uint8)
        print('Saving npz to %s...' % npz_filename)
        np.savez(npz_filename, **{'x': X})
                
   

    #=====Classify===================================================================
    # metrics = {'l2': 0, 'l1': 0, 'kl': 0}
    # l2_db = np.zeros(10)
    # l1_db = np.zeros(10)
    # kl_db = np.zeros(10)

    # output file
    # fname = '%s/%s_fair_disc_fid_samples.p' % (config['samples_root'], perc_bias)

    # load classifier 
    # #(Saved state)
    # if not config['multi']:
    #     # print('Pre-loading pre-trained single-attribute classifier...')
    #     # clf_state_dict = torch.load(CLF_PATH)['state_dict']
    #     clf_classes = attributes
    # else:
    #     # multi-attribute
    #     # print('Pre-loading pre-trained multi-attribute classifier...')
    #     # clf_state_dict = torch.load(MULTI_CLF_PATH)['state_dict']
    #     clf_classes = attributes
        
    # # load attribute classifier here
    # #(Model itself)
    # # clf = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=clf_classes, grayscale=False) 
    # # clf.load_state_dict(clf_state_dict)
    # # clf = Net(clf_classes) 
    # # clf.load_state_dict(clf_state_dict)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # clf = clf.to(device)
    # clf.eval()  # turn off batch norm

    # classify examples and get probabilties
    # n_classes = 2
    # if config['multi']:
    #     n_classes = 4

    # number of classes
    # probs_db = np.zeros((1, config['sample_num_npz'], n_classes)) #Numper of runs , images per run ,Number of classes
    # for i in range(1):
        # grab appropriate samples
        # npz_filename = '%s/%s_fid_real_samples_%s.npz' % ("./samples", perc_bias, k) #E.g. perc_fid_samples_0
        # preds, probs = classify_examples(clf, npz_filename) #Classify the data
        
        # l2, l1, kl = utils.fairness_discrepancy(preds, clf_classes) #Pass to calculate score
        
        #exp
        # l2Exp, l1Exp, klExp = utils.fairness_discrepancy_exp(probs, clf_classes) #Pass to calculate score

        # save metrics (To add on new mertrics add here)
        # l2_db[i] = l2
        # l1_db[i] = l1
        # kl_db[i] = kl
        # probs_db[i] = probs
        
        #Write log
        # f.write("Running: "+npz_filename+"\n")
        # f.write('fair_disc for iter {} is: l2:{}, l1:{}, kl:{} \n'.format(i, l2, l1, kl))
        
        
        # print('fair_disc for iter {} is: l2:{}, l1:{}, kl:{}'.format(i, l2, l1, kl))
        # print('fair_disc_exp for iter {} is: l2:{}, l1:{}, kl:{} \n'.format(i, l2Exp, l1Exp, klExp))
        
        
        #Commented out for stamford experiment*************************************************************************************************
        #FID score 50_50 vs others 
        # data_moments=os.path.join("./samples","0.5_fid_real_samples_ref_0.npz")
        # sample_moments=os.path.join("./samples",'%s_fid_real_samples_%s.npz'%(perc_bias,k))
        # FID = fid_score_mod.calculate_fid_given_paths([data_moments, sample_moments], batch_size=100, cuda=True, dims=2048)
        # FID = fid_score_mod_AE.calculate_faed_given_paths([data_moments, sample_moments], batch_size=100, cuda=True, dims=2048)

        # print ("FID: "+str(FID))
        # f.write("FID: "+str(FID)+"\n")     
        #Commented out for stamford experiment*************************************************************************************************
        
        # f.close()
    # metrics['l2'] = l2_db
    # metrics['l1'] = l1_db
    # metrics['kl'] = kl_db
    # print('fairness discrepancies saved in {}'.format(fname))
    # print(l2_db)
    
    # save all metrics
    # with open(fname, 'wb') as fp:
    #     pickle.dump(metrics, fp)
    # np.save(os.path.join(config['samples_root'], 'clf_probs.npy'), probs_db)


def main():
    # parse command line and run
    # parser = utils.prepare_parser()
    # parser = utils.add_sample_parser(parser)
    # config = vars(parser.parse_args())
    # print(config)
    run()


if __name__ == '__main__':
    main()
