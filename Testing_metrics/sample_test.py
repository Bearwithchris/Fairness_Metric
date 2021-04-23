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
import argparse

# Import my stuff
# import inception_utils
# import utils
# import utils_add_on as uao
# import losses
# from clf_models import ResNet18, BasicBlock, Net

import sys
sys.path.append('../Data_prep')
from clf_models import ResNet18, BasicBlock, Net


CLF_PATH = '../Data_prep/results/attr_clf/model_best.pth.tar'
MULTI_CLF_PATH = '../Data_prep/results/multi_clf/model_best.pth.tar'

def fairness_discrepancy(data, n_classes):
    """
    computes fairness discrepancy metric for single or multi-attribute
    this metric computes L2, L1, AND KL-total variation distance
    """
    unique, freq = np.unique(data, return_counts=True)
    props = freq / len(data) #Proportion of data that belongs to that data
    print (freq)
    truth = 1./n_classes


    # L2 and L1
    l2_fair_d = np.sqrt(((props - truth)**2).sum())
    l1_fair_d = abs(props - truth).sum()

    # q = props, p = truth
    kl_fair_d = (props * (np.log(props) - np.log(truth))).sum()

    #Cross entropy
    p=np.ones(n_classes)/n_classes    
    # ce=cross_entropy(p,props,n_classes)-cross_entropy(p,p,n_classes)
    
    #information specificity
    rank=np.linspace(1,n_classes-1,n_classes-1)
    rank[::-1].sort() #Descending order
    perc=np.array([i/np.sum(rank) for i in rank])
    props[::-1].sort()
    alpha=props[1:]
    specificity=props[0]-np.sum(alpha*perc)
    info_spec=(l1_fair_d+specificity)/2
    
    
    return l2_fair_d, l1_fair_d, kl_fair_d


def load_data(attributes,index):
    data=torch.load('../data/resampled_ratio/gen_data_%i_%s'%(attributes,index))
    print ("Data loaded: "+'../data/resampled_ratio/gen_data_%i_%s'%(attributes,index))
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
    remainder=samples.shape[0]-(n_batches*1000)
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
            
        if remainder!=0:
            x = samples[(i+1)*1000:(1000*(i+1))+remainder]
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
    parser.add_argument('--index', type=int, help='dataset index to load', default=1)
    parser.add_argument('--class_idx', type=int, help='CelebA class label for training.', default=20)
    parser.add_argument('--multi_class_idx',nargs="*", type=int, help='CelebA class label for training.', default=[6,7,8])
    parser.add_argument('--multi', type=bool, default=True, help='If True, runs multi-attribute classifier')
    parser.add_argument('--split_type', type=str, help='[train,val,split]', default="test")
    args = parser.parse_args()
    # state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num_fair': 0, 'save_best_num_fid': 0, 'best_IS': 0, 'best_FID': 999999, 'best_fair_d': 999999, 'config': config}

    # Optionally, get the configuration from the state dict. This allows for
    # recovery of the config provided only a state dict and experiment name,
    # and can be convenient for writing less verbose sample shell scripts.
    
    # if config['config_from_name']: #Default is false
    #     utils.load_weights(None, None, state_dict, config['weights_root'],
    #                        config['experiment_name'], config['load_weights'], None,
    #                        strict=False, load_optim=False)
    #     # Ignore items which we might want to overwrite from the command line
    #     for item in state_dict['config']:
    #         if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
    #             config[item] = state_dict['config'][item]

    # update config (see train.py for explanation)
    # config['resolution'] = utils.imsize_dict[config['dataset']]
    # config['n_classes'] = 1
    
    # if config['conditional']:
    #     config['n_classes'] = 2
    # config['G_activation'] = utils.activation_dict[config['G_nl']]
    # config['D_activation'] = utils.activation_dict[config['D_nl']]
    # config = utils.update_config_roots(config)
    # config['skip_init'] = True
    # config['no_optim'] = True
    device = 'cuda'
    # config['sample_num_npz'] = 10000
    # config['sample_num_npz'] = 10000
    # perc_bias=float(config["bias"].split("_")[0])/100
    # print(config['ema_start'])

    # Seed RNG
    # utils.seed_rng(config['seed'])  # config['seed'])

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True


    #Log Runs
    f=open('../%s/log_stamford_fair.txt' %("logs"),"a")

    # experiment_name = (config['experiment_name'] if config['experiment_name'] #Default CelebA
    #                     else utils.name_from_config(config))
    
    #Load dataset to be tested
    if args.multi:
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
    npz_filename = '%s/%s_fid_real_samples_%s.npz' % (file,attributes, args.index) #E.g. perc_fid_samples_0
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
    metrics = {'l2': 0, 'l1': 0, 'kl': 0}
    l2_db = np.zeros(10)
    l1_db = np.zeros(10)
    kl_db = np.zeros(10)

    # output file
    # fname = '%s/%s_fair_disc_fid_samples.p' % (config['samples_root'], perc_bias)

    # load classifier 
    #(Saved state)
    if not args.multi:
        print('Pre-loading pre-trained single-attribute classifier...')
        clf_state_dict = torch.load(CLF_PATH)['state_dict']
        clf_classes = attributes
    else:
        # multi-attribute
        print('Pre-loading pre-trained multi-attribute classifier...')
        clf_state_dict = torch.load(MULTI_CLF_PATH)['state_dict']
        clf_classes = attributes
        
    # load attribute classifier here
    #(Model itself)
    clf = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=attributes, grayscale=False) 
    clf.load_state_dict(clf_state_dict)
    # clf = Net(clf_classes) 
    # clf.load_state_dict(clf_state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = clf.to(device)
    clf.eval()  # turn off batch norm

    # classify examples and get probabilties
    # n_classes = 2
    # if config['multi']:
    #     n_classes = 4

    # number of classes
    probs_db = np.zeros((1, size, clf_classes)) #Numper of runs , images per run ,Number of classes
    for i in range(1):
        # grab appropriate samples
        npz_filename = os.path.join("../data","FID_sample_storage_%i"%attributes,'%s_fid_real_samples_%s.npz' % (attributes, args.index))
        preds, probs = classify_examples(clf, npz_filename) #Classify the data
        
        l2, l1, kl = fairness_discrepancy(preds, clf_classes) #Pass to calculate score
        
        #exp
        # l2Exp, l1Exp, klExp = utils.fairness_discrepancy_exp(probs, clf_classes) #Pass to calculate score

        # save metrics (To add on new mertrics add here)
        l2_db[i] = l2
        l1_db[i] = l1
        kl_db[i] = kl
        probs_db[i] = probs
        
        #Write log
        # f.write("Running: "+npz_filename+"\n")
        f.write('fair_disc for iter {} is: l2:{}, l1:{}, kl:{} \n'.format(i, l2, l1, kl))
        
        
        print('fair_disc for iter {} is: l2:{}, l1:{}, kl:{}'.format(i, l2, l1, kl))
        # print('fair_disc_exp for iter {} is: l2:{}, l1:{}, kl:{} \n'.format(i, l2Exp, l1Exp, klExp))
        
        
        #Commented out for stamford experiment*************************************************************************************************
        # #FID score 50_50 vs others 
        # data_moments=os.path.join("./samples","0.5_fid_real_samples_ref_0.npz")
        # sample_moments=os.path.join("./samples",'%s_fid_real_samples_%s.npz'%(perc_bias,k))
        # # FID = fid_score_mod.calculate_fid_given_paths([data_moments, sample_moments], batch_size=100, cuda=True, dims=2048)
        # FID = fid_score_mod_AE.calculate_faed_given_paths([data_moments, sample_moments], batch_size=100, cuda=True, dims=2048)

        # print ("FID: "+str(FID))
        # f.write("FID: "+str(FID)+"\n")     
        #Commented out for stamford experiment*************************************************************************************************
        
        f.close()
    metrics['l2'] = l2_db
    metrics['l1'] = l1_db
    metrics['kl'] = kl_db
    # print('fairness discrepancies saved in {}'.format(fname))
    print(l2_db)
    
    # # save all metrics
    # with open(fname, 'wb') as fp:
    #     pickle.dump(metrics, fp)
    # np.save(os.path.join(config['samples_root'], 'clf_probs.npy'), probs_db)


# def main():
#     # parse command line and run
#     parser = utils.prepare_parser()
#     parser = utils.add_sample_parser(parser)
#     config = vars(parser.parse_args())
#     print(config)
#     run(config)


if __name__ == '__main__':
    run()
#     main()
