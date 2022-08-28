# inference demo (need a GPU)
python inference.py --process-stack-root data/dataset/raw --json-store-root data/dataset_result

# or you can infer data with human-proofreading pre-precessing results
#python inference.py --process-stack-root data/dataset/raw --json-store-root data/dataset_result --load-preprocess-result-root data/dataset/proofreading

# if you want to generate qualitative results, please add --store-vis-result (it will consume too much memory)
#python inference.py --process-stack-root data/dataset/raw --json-store-root data/dataset_result --load-preprocess-result-root data/dataset/proofreading --store-vis-result