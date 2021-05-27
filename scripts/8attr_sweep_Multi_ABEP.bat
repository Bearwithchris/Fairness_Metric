cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8 31
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8 31
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8 31

cd ../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --mode_normal=2 --ABEP=0 --multi=1  --multi_class_idx 20 8 31
cd ../Testing_metrics
for /l %%x in (0, 1, 181) do python sample_test.py --multi_clf_path=multi_clf_8 --multi=1 --multi_class_idx 20 8 31 --index=%%x


cd ../data/FID_sample_storage_8
del *.npz


cd ../../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --mode_normal=2 --ABEP=1 --multi=1  --multi_class_idx 20 8 31
cd ../Testing_metrics
for /l %%x in (0, 1, 181) do python sample_test.py --multi_clf_path=multi_clf_8 --multi=1 --multi_class_idx 20 8 31 --index=%%x


cd ../data/FID_sample_storage_8
del *.npz

cd ../../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --mode_normal=2 --ABEP=2 --multi=1  --multi_class_idx 20 8 31
cd ../Testing_metrics
for /l %%x in (0, 1, 181) do python sample_test.py --multi_clf_path=multi_clf_8 --multi=1 --multi_class_idx 20 8 31 --index=%%x

cd ../data/FID_sample_storage_8
del *.npz


cd ../../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --mode_normal=2 --ABEP=1 --multi=3  --multi_class_idx 20 8 31
cd ../Testing_metrics
for /l %%x in (0, 1, 181) do python sample_test.py --multi_clf_path=multi_clf_8 --multi=1 --multi_class_idx 20 8 31 --index=%%x

cd ../data/FID_sample_storage_8
del *.npz


cd ../../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --mode_normal=2 --ABEP=1 --multi=4  --multi_class_idx 20 8 31
cd ../Testing_metrics
for /l %%x in (0, 1, 181) do python sample_test.py --multi_clf_path=multi_clf_8 --multi=1 --multi_class_idx 20 8 31 --index=%%x

cd ../data/FID_sample_storage_8
del *.npz

cd ../../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --mode_normal=2 --ABEP=1 --multi=5  --multi_class_idx 20 8 31
cd ../Testing_metrics
for /l %%x in (0, 1, 181) do python sample_test.py --multi_clf_path=multi_clf_8 --multi=1 --multi_class_idx 20 8 31 --index=%%x


cd ../data/FID_sample_storage_8
del *.npz


cd ../../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --mode_normal=2 --ABEP=6 --multi=1  --multi_class_idx 20 8 31
cd ../Testing_metrics
for /l %%x in (0, 1, 181) do python sample_test.py --multi_clf_path=multi_clf_8 --multi=1 --multi_class_idx 20 8 31 --index=%%x


cd ../data/FID_sample_storage_8
del *.npz


cd ../../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --mode_normal=2 --ABEP=7 --multi=1  --multi_class_idx 20 8 31
cd ../Testing_metrics
for /l %%x in (0, 1, 181) do python sample_test.py --multi_clf_path=multi_clf_8 --multi=1 --multi_class_idx 20 8 31 --index=%%x


