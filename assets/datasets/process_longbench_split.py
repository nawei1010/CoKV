import sys
project_path=
sys.path.append(project_path)
import os
import json
from datasets import load_dataset
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--dataset', type=str, default="path_to_your_longbench/LongBench")
    return parser.parse_args(args)


def process_and_save_data(dataset, data, out_valid_path, out_test_path):
    data = data.shuffle(seed=42) 
    length = 30
    if len(data) > 200:
        length = 50
    valid_data = data.select(range(length))  
    test_data = data.select(range(length, len(data)))

    valid_file_path = os.path.join(out_valid_path, f"{dataset}.jsonl")
    test_file_path = os.path.join(out_test_path, f"{dataset}.jsonl")

    valid_data.to_json(valid_file_path)
    test_data.to_json(test_file_path)



if __name__ == '__main__':
    args = parse_args()

    datasets = ["qasper", "multi_news", "narrativeqa", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                "gov_report", "qmsum", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en",  \
                "lcc", "repobench-p"]

    out_valid_path = f'{project_path}/assets/datasets/LongBench-valid'
    out_test_path = f'{project_path}/assets/datasets/LongBench-test'
    
    os.makedirs(out_valid_path, exist_ok=True)
    os.makedirs(out_test_path, exist_ok=True)

    for dataset in datasets:
        data = load_dataset('json', data_files={'test': f"{args.dataset}/{dataset}.jsonl"}, split='test')
        process_and_save_data(dataset, data, out_valid_path, out_test_path)
