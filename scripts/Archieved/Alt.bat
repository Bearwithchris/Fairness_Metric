cd ../Testing_metrics
for /l %%x in (0, 1, 63) do python sample_test.py --multi=True --multi_class_idx 20 8 7 6 --index=%%x