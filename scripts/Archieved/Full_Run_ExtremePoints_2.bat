cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20
python train_attribute_clf.py celeba ./results/multi_clf --even=1 --multi=1 --multi_class_idx 20
python validate_acc.py celeba ./results/multi_clf --even=1 --multi=1 --multi_class_idx 20
cd ../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --mode_normal=0 --split_type test --multi=1  --multi_class_idx 20
cd ../Testing_metrics
for /l %%x in (0, 1, 2) do python sample_test.py --multi=1 --multi_class_idx 20 --index=%%x

cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8
python train_attribute_clf.py celeba ./results/multi_clf --even=1 --multi=1 --multi_class_idx 20 8
python validate_acc.py celeba ./results/multi_clf --even=1 --multi=1 --multi_class_idx 20 8
cd ../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --mode_normal=0 --split_type test --multi=1  --multi_class_idx 20 8
cd ../Testing_metrics
for /l %%x in (0, 1, 4) do python sample_test.py --multi=1 --multi_class_idx 20 8 --index=%%x

cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8 31
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8 31
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8 31
python train_attribute_clf.py celeba ./results/multi_clf --even=1 --multi=1 --multi_class_idx 20 8 31
python validate_acc.py celeba ./results/multi_clf --even=1 --multi=1 --multi_class_idx 20 8 31
cd ../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --mode_normal=0 --split_type test --multi=1  --multi_class_idx 20 8 31
cd ../Testing_metrics
for /l %%x in (0, 1, 8) do python sample_test.py --multi=1 --multi_class_idx 20 8 31 --index=%%x

cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8 31 5
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8 31 5
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8 31 5
python train_attribute_clf.py celeba ./results/multi_clf --even=1 --multi=1 --multi_class_idx 20 8 31 5
python validate_acc.py celeba ./results/multi_clf --even=1 --multi=1 --multi_class_idx 20 8 31 5
cd ../CelebA_data_split
python .\gen_celebA_dataset.py --multi=1 --mode_normal=0 --split_type test --multi=1  --multi_class_idx 20 8 31 5
cd ../Testing_metrics
for /l %%x in (0, 1, 16) do python sample_test.py --multi=1 --multi_class_idx 20 8 31 5 --index=%%x