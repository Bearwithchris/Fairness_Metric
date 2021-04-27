cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20
python train_attribute_clf.py celeba ./results/multi_clf --even=1 --multi=1 --multi_class_idx 20
cd ../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --multi=1  --multi_class_idx 20
cd ../Testing_metrics
for /l %%x in (0, 1, 1) do python sample_test.py --multi=1 --multi_class_idx 20 --index=%%x

cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8
python train_attribute_clf.py celeba ./results/multi_clf --even=1 --multi=1 --multi_class_idx 20 8
cd ../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --multi=1  --multi_class_idx 20 8
cd ../Testing_metrics
for /l %%x in (0, 1, 4) do python sample_test.py --multi=1 --multi_class_idx 20 8 --index=%%x

cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8 7
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8 7
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8 7
python train_attribute_clf.py celeba ./results/multi_clf --even=1 --multi=1 --multi_class_idx 20 8 7
cd ../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --multi=1  --multi_class_idx 20 8 7
cd ../Testing_metrics
for /l %%x in (0, 1, 16) do python sample_test.py --multi=1 --multi_class_idx 20 8 7 --index=%%x

cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8 7 6
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8 7 6
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8 7 6
python train_attribute_clf.py celeba ./results/multi_clf --even=1 --multi=1 --multi_class_idx 20 8 7 6
cd ../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --split_type test --multi=1  --multi_class_idx 20 8 7 6
cd ../Testing_metrics
for /l %%x in (0, 1, 63) do python sample_test.py --multi=1 --multi_class_idx 20 8 7 6 --index=%%x