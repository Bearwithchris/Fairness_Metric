cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 39
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 39
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 39
python train_attribute_clf.py celeba ./results/multi_clf --multi=True --even=True --multi_class_idx 39
cd ../CelebA_data_split
python .\gen_celebA_dataset.py --multi=True --split_type test --multi=True  --multi_class_idx 39
cd ../Testing_metrics
for /l %%x in (0, 1, 1) do python sample_test.py --multi=True --multi_class_idx 39 --index=%%x