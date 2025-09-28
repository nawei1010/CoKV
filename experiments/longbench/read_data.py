import json
import argparse

ORDERED_DATASETS = [
    "narrativeqa",          # NtrQA
    "qasper",               # Qasper
    "multifieldqa_en",      # MF-en
    "hotpotqa",             # HotpotQA
    "2wikimqa",             # 2WikiMQA
    "musique",              # Musique
    "gov_report",           # GovReport
    "qmsum",                # QMSum
    "multi_news",           # MultiNews
    "trec",                 # TREC
    "triviaqa",             # TriviaQA
    "samsum",               # SAMSum
    "passage_count",        # PCount
    "passage_retrieval_en", # PRe
    "lcc",                  # Lcc
    "repobench-p",          # RB-P
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=".//experiments/longbench/pred_now/result_qwen3_test_speed_20_ada_1024_alpha_0.2_gqa_support_True.json",
        help="{dataset: number}"
    )
    parser.add_argument("--no-backslash", action="store_true", help="\\\")
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for ds in ORDERED_DATASETS:
        v = data.get(ds)
        s = str(v) if isinstance(v, (int, float)) else ""
        line = f"& {s}"
        
        print(line)

if __name__ == "__main__":
    main()