cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8
python train_attribute_clf.py celeba ./results/multi_clf --multi=1 --even=1 --multi_class_idx 20 8
cd ../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --mode_normal=0 --multi=1  --multi_class_idx 20 8
cd ../Testing_metrics
for /l %%x in (0, 1, 4) do python sample_test.py --multi=1 --multi_clf_path=multi_clf_4  --multi_class_idx 20 8 --index=%%x

cd ../data/FID_sample_storage_4
del *.npz

cd ../../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 39 31
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 39 31
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 39 31
python train_attribute_clf.py celeba ./results/multi_clf --multi=1 --even=1 --multi_class_idx 39 31
cd ../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --mode_normal=0 --multi=1  --multi_class_idx 39 31
cd ../Testing_metrics
for /l %%x in (0, 1, 4) do python sample_test.py --multi=1 --multi_clf_path=multi_clf_4  --multi_class_idx 39 31 --index=%%x
