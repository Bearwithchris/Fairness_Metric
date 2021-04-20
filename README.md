## 1) Data setup:
(a) Download the CelebA dataset here (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) into the `data/` directory (if elsewhere, note the path for step b). Of the download links provided, choose `Align&Cropped Images` and download `Img/img_align_celeba/` folder, `Anno/list_attr_celeba.txt`, and `Eval/list_eval_partition.txt` to `data/`.

(b) Preprocess the CelebA dataset for faster training:
>Split into train/test/val
```
python preprocess_celeba.py --data_dir=E:\GIT\local_data_in_use\CelebA_data --out_dir=../data --partition=train
```

To split the data for multiple attributes
```
python .\preprocess_celeba_multi.py --split_type=train --class_idx 8 20
```



## 2) Pre-train attribute classifier
For a single-attribute:
```
python3 train_attribute_clf.py celeba ./results/attr_clf
```

For multiple attributes, add the `--multi=True` flag.
