cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 2
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 2
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 2
python train_attribute_clf.py celeba ./results/multi_clf_2 --even=1 --multi=1 --multi_class_idx 2
python validate_acc.py celeba ./results/multi_clf_2 --even=1 --multi=1 --multi_class_idx 2
