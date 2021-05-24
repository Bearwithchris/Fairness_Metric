cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8 31
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8 31
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8 31

cd ../CelebA_data_split
python .\gen_celebA_dataset.py --step_mul=2 --multi=1 --split_type test --mode_normal=2 --multi=1  --multi_class_idx 20 8 31
cd ../Testing_metrics
for /l %%x in (0, 1, 181) do python sample_test.py --multi_clf_path=multi_clf_8 --multi=1 --multi_class_idx 20 8 31 --index=%%x

