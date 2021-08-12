import os
import sys
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from clf_models import BasicBlock, build_model, Net
from utils import save_checkpoint
from dataset_splits import (
    build_celeba_classification_dataset,
    build_multi_celeba_classification_datset,
    build_custom_celeba_classification_dataset,
    build_custom_multi_celeba_classification_datset,
    build_multi_even_celeba_classification_datset,
)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default="celeba", help='celeba')
    parser.add_argument('out_dir', type=str,default='./results/multi_clf_2/model_best.pth.tar' ,help='where to save outputs')
    parser.add_argument('--ckpt-path', type=str, default='./results/multi_clf', 
                        help='if test=True, path to clf checkpoint')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='minibatch size [default: 64]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs [default: 10]')
    parser.add_argument('--class_idx', type=int, default=20,
                        help='CelebA class label for training.')
    parser.add_argument('--multi_class_idx',nargs="*", type=int, help='CelebA class label for training.',default=[20]) #default=[20,8,7,6])
    parser.add_argument('--multi', type=int, default=1, 
                        help='If True, runs multi-attribute classifier')
    parser.add_argument('--even', type=int, default=1, 
                        help='If True, runs multi-even-attribute classifier')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    
    #Simulate
    # argv = ["celeba ","./results/multi_clf "]
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # reproducibility
    torch.manual_seed(777)
    np.random.seed(777)

    device = torch.device('cuda' if args.cuda else 'cpu')
    
      #Create output folder
    # if not os.path.isdir(args.out_dir):
        # os.makedirs(args.out_dir)

    # get data: idx 20 = male, idx 8 = black hair
    if args.multi==0:
        train_dataset = build_celeba_classification_dataset(
              'train', args.class_idx)
        # train_dataset = build_custom_celeba_classification_dataset(
             # 'train', args.class_idx,count=args.count)
        valid_dataset = build_celeba_classification_dataset(
            'val', args.class_idx)
        test_dataset = build_celeba_classification_dataset(
            'test', args.class_idx)
        n_classes = 2
        CLF_PATH = '../Data_prep/results/attr_clf/model_best.pth.tar'

    else:
        # CLF_PATH = '../Data_prep/results/multi_clf/model_best.pth.tar'
        CLF_PATH = args.out_dir
        if args.even==0:
            # (Dataset are already in torch.utils.data format)
            # train_dataset = build_multi_celeba_classification_datset('train')
            # train_dataset = build_custom_multi_celeba_classification_datset('train',args.count)
            # valid_dataset = build_multi_celeba_classification_datset('val')
            test_dataset = build_multi_celeba_classification_datset('test')
        else:
            # train_dataset = build_multi_even_celeba_classification_datset('train')
            # train_dataset = build_custom_multi_celeba_classification_datset('train',args.count)
            # valid_dataset = build_multi_even_celeba_classification_datset('val')
            test_dataset = build_multi_even_celeba_classification_datset('test')
            
        n_classes =  2**len(args.multi_class_idx)

    f=open("../logs/Attribute_classifier_accuracy.txt","a")
    f.write("Testing on test sample size: "+str(len(test_dataset))+" and attributes "+str(args.multi_class_idx)+"\n")
    print(len(test_dataset))

    # train/validation split (Shuffle and batch the datasets)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)


    def test(loader):
        model.eval()
        test_loss = 0
        correct = 0
        num_examples = 0
        predictedArray=[]
        targetArray=[]
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                data = data.float() / 255.
                target = target.long()

                # run through model
                logits, probas = model(data)
                # print (logits[0])
                # print (target[0])
                # test_loss += F.cross_entropy(logits, target, reduction='sum').item() # sum up batch loss
                _, pred = torch.max(probas, 1)
                # num_examples += target.size(0)
                # correct += (pred == target).sum()
                predictedArray.append(pred.cpu().numpy())
                targetArray.append(target.cpu().numpy())
            predicted=np.concatenate(predictedArray)
            target=np.concatenate(targetArray)
            correct=(predicted==target)
            scores=np.zeros(len(np.unique(target)))
            for index in range(len(np.unique(target))):
                position=np.where(target==index)[0]
                scores[index]=correct[position].sum()/len(position)
                  
        for i in range(len(scores)):
            f.write(" Attribute_%i="%i+str(scores[i]))
        f.write("\n")
        # f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, num_examples,100. * correct / num_examples))
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, num_examples,
        #     100. * correct / num_examples))
        return 0

    # classifier has finished training, evaluate sample diversity
    best_loss = sys.maxsize
    clf_state_dict = torch.load(CLF_PATH)['state_dict']
 
    # reload best model
    model_cls = build_model('celeba')
    model = model_cls(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=n_classes, grayscale=False)
    model = model.to(device)
     # model=Net(n_classes)
     # model.cuda()
    model.load_state_dict(clf_state_dict)

    # get test
    test_loss = test(test_loader)
    f.close()