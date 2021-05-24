cd ../Data_prep
python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20
python train_attribute_clf.py celeba ./results/multi_clf_2 --even=1 --multi=1 --multi_class_idx 20
python validate_acc.py celeba ./results/multi_clf_2 --even=1 --multi=1 --multi_class_idx 20

python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8
python train_attribute_clf.py celeba ./results/multi_clf_4 --even=1 --multi=1 --multi_class_idx 20 8
python validate_acc.py celeba ./results/multi_clf_4 --even=1 --multi=1 --multi_class_idx 20 8


python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8 31
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8 31
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8 31
python train_attribute_clf.py celeba ./results/multi_clf_8 --even=1 --multi=1 --multi_class_idx 20 8 31
python validate_acc.py celeba ./results/multi_clf_8 --even=1 --multi=1 --multi_class_idx 20 8 31


python .\preprocess_celeba_multi.py --split_type=train --multi_class_idx 20 8 31 5
python .\preprocess_celeba_multi.py --split_type=test --multi_class_idx 20 8 31 5
python .\preprocess_celeba_multi.py --split_type=val --multi_class_idx 20 8 31 5
python train_attribute_clf.py celeba ./results/multi_clf_16 --even=1 --multi=1 --multi_class_idx 20 8 31 5
python validate_acc.py celeba ./results/multi_clf_16 --even=1 --multi=1 --multi_class_idx 20 8 31 5
