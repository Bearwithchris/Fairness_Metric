cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8

cd ../CelebA_data_split
python .\gen_celebA_dataset.py --step_mul=4 --multi=1 --split_type test --mode_normal=2 --ABEP=0 --multi=1  --multi_class_idx 20 8
cd ../Testing_metrics
for /l %%x in (0, 1, 77) do python sample_test.py --multi_clf_path=multi_clf_4 --multi=1 --multi_class_idx 20 8 --index=%%x

cd ../data/FID_sample_storage_4
del *.npz

cd ../../CelebA_data_split
python .\gen_celebA_dataset.py --step_mul=4 --multi=1 --split_type test --mode_normal=2 --ABEP=1 --multi=1  --multi_class_idx 20 8
cd ../Testing_metrics
for /l %%x in (0, 1, 77) do python sample_test.py --multi_clf_path=multi_clf_4 --multi=1 --multi_class_idx 20 8 --index=%%x

cd ../data/FID_sample_storage_4
del *.npz

cd ../../CelebA_data_split
python .\gen_celebA_dataset.py --step_mul=4 --multi=1 --split_type test --mode_normal=2 --ABEP=3 --multi=1  --multi_class_idx 20 8
cd ../Testing_metrics
for /l %%x in (0, 1, 77) do python sample_test.py --multi_clf_path=multi_clf_4 --multi=1 --multi_class_idx 20 8 --index=%%x

cd ../data/FID_sample_storage_4
del *.npz

cd ../../CelebA_data_split
python .\gen_celebA_dataset.py --step_mul=4 --multi=1 --split_type test --mode_normal=2 --ABEP=4 --multi=1  --multi_class_idx 20 8
cd ../Testing_metrics
for /l %%x in (0, 1, 77) do python sample_test.py --multi_clf_path=multi_clf_4 --multi=1 --multi_class_idx 20 8 --index=%%x