## 1) Data setup:
(a) Download the CelebA dataset here (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) into the `data/` directory (if elsewhere, note the path for step b). Of the download links provided, choose `Align&Cropped Images` and download `Img/img_align_celeba/` folder, `Anno/list_attr_celeba.txt`, and `Eval/list_eval_partition.txt` to `data/`.
1-2: Scripts can be found in Data_prep folder
3: Scrips can be found in CelebA_data_split
(b) Preprocess the CelebA dataset for faster training:
**** Remember to do it for all 3 catergories ****
(In Data_prep folder)
>Split into train/test/val
```
python preprocess_celeba.py --data_dir=E:\GIT\local_data_in_use\CelebA_data --out_dir=../data --partition=train
```

To split the data for multiple attributes
```
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 8 20
```



## 2) Pre-train attribute classifier 
(In Data_prep folder)
For a single-attribute:
```
python train_attribute_clf.py celeba ./results/multi_clf_2 --even=1 --multi=1 --multi_class_idx 20
```
For multiple attributes
``` 
python train_attribute_clf.py celeba ./results/multi_clf_4 --even=1 --multi=1 --multi_class_idx 20 8
```


## 3) Prep Testing data 
(In CelebA_data_split folder)
> Segment the data into decrease distribution of long-tail to even distriubtions of data for testing
>Exreteme points, mode_normal=0. Sweep, mode_normal=2
```
python .\gen_celebA_dataset.py --multi=1 --split_type test --mode_normal=2 --multi=1  --multi_class_idx 20 8
```
>Extreme point test
```
python .\gen_celebA_dataset.py --multi=1 --split_type test --mode_normal=0 --multi=1  --multi_class_idx 20 8
```
>Validating the individual accuracies
```
python validate_acc.py celeba ./results/multi_clf_8 --even=1 --multi=1 --multi_class_idx 20 8
```

## 4) Running Test 
(In Testing_metrics folder)
>Run the dist test
>Define the index you would want to run on, with reference to the gen_celebA_dataset indexing
```
python sample_test.py --multi_clf_path=multi_clf_4 --multi=1 --multi_class_idx 20 8 --index=%%x
```


## Additional experiments
(A) Random sweep -> Experimental Upper bound
>1) Run .\scripts\random_Sweep to generate the psuedo random distribution and classified distribution (Assumed that the classifier has been trained with Train_2_4_8_16, chanve the multi_class_idx to the respective)
>2) Move .\logs\ideal_dist.npz and .\logs\pred_dist.npz to .\Testing_metrics\Experiment_error\A_t_x x={2,4,8,16}
>3) Run the error.py script

## Auto scrips
>scripts are under ./scripts file
>Accuracy varaibility scrips, note that the classifiers have to be trained each round since they are testing on different attributes. These classifiers do not clash with Train_2_4_8_16 which saves the classifier in a different file
```
Accuracy_Single_Attribute_Run_ep
Accuracy_double_Attribute_Run_ep
```
>Train classifiers
```
Train_2_4_8_16
```
>Sweeping distribution 
```
[2,4,8]attr_sweep
```
>Run All Extreme Poitns for all attributes
```
Full_Run_ExtremePoints
```
>Clean start, deletes all the cahced files
```
Clean
```