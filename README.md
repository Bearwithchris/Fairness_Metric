## 1) Data setup:
(a) Download the CelebA dataset here (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) into the `data/` directory (if elsewhere, note the path for step b). Of the download links provided, choose `Align&Cropped Images` and download `Img/img_align_celeba/` folder, `Anno/list_attr_celeba.txt`, and `Eval/list_eval_partition.txt` to `data/`.
1-2: Scripts can be found in Data_prep folder
3: Scrips can be found in CelebA_data_split
(b) Preprocess the CelebA dataset for faster training:
**** Remember to do it for all 3 catergories ****
>Split into train/test/val
```
python preprocess_celeba.py --data_dir=E:\GIT\local_data_in_use\CelebA_data --out_dir=../data --partition=train
```

To split the data for multiple attributes
```
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 8 20
```



## 2) Pre-train attribute classifier
For a single-attribute:
```
python3 train_attribute_clf.py celeba ./results/attr_clf --class_idx 20
```
For multiple attributes
``` 
python train_attribute_clf.py celeba ./results/multi_clf --multi=True --multi_class_idx 6 7 8 20
```
For multiple attributes with even training data
``` 
python train_attribute_clf.py celeba ./results/multi_clf --multi=True --even=True --multi_class_idx 6 7 8 20
```

## 3) Prep Testing data 
> Segment the data into decrease distribution of long-tail to even distriubtions of data for testing
```
python .\gen_celebA_dataset.py --multi=True --mode_normal==True --split_type test --multi=True  --multi_class_idx 6 7 8 20
```
>Extreme point test
```
python .\gen_celebA_dataset.py --multi=True --mode_normal==False --split_type test --multi=True  --multi_class_idx 6 7 8 20
```

## 4) Running Test 
>Run the dist test
```
python sample_test.py --multi=True --multi_class_idx 6 7 8 20
```
