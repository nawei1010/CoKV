import os
import json
from typing import Dict, Any, List

try:
	from datasets import load_from_disk
except Exception:
	load_from_disk = None


def _split_rationale_and_answer(raw_answer: str) -> Dict[str, str]:

	if raw_answer is None:
		return {"rationale": "", "final": ""}
	answer = raw_answer
	parts = answer.split('####')
	if len(parts) >= 2:
		rationale = parts[0].strip()
		final = parts[-1].strip()
	else:
		rationale = answer.strip()
		final = ""
	# Clean final numeric
	final = final.replace(',', '')
	if final.endswith('.'):
		final = final[:-1]
	return {"rationale": rationale, "final": final.strip()}


def _normalize_gsm8k_answer(raw_answer: str) -> str:

	if raw_answer is None:
		return ""
	answer = raw_answer
	if '####' in answer:
		answer = answer.split('####')[-1]
	answer = answer.strip()
	# Remove trailing period and commas/spaces for consistency
	answer = answer.replace(',', '')
	if answer.endswith('.'):
		answer = answer[:-1]
	return answer.strip()


def _write_jsonl(path: str, rows: List[Dict[str, Any]]):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _file_has_rationale(path: str) -> bool:
	try:
		with open(path, 'r', encoding='utf-8') as f:
			for i, line in enumerate(f):
				if i > 10:
					break
				obj = json.loads(line)
				if 'rationale' in obj:
					return True
		return False
	except Exception:
		return False


def prepare_gsm8k_dataset(dest_root: str, valid_size: int = 50, seed: int = 42, rebuild: bool = False) -> Dict[str, str]:

	out_dir = os.path.join(dest_root, 'gsm8k')
	train_path = os.path.join(out_dir, 'train.jsonl')
	valid_path = os.path.join(out_dir, 'valid.jsonl')
	test_path = os.path.join(out_dir, 'test.jsonl')

	# If already prepared, optionally rebuild if rationale missing or rebuild flag set
	if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path) and not rebuild:
		if _file_has_rationale(train_path) and _file_has_rationale(valid_path) and _file_has_rationale(test_path):
			return {"train": train_path, "valid": valid_path, "test": test_path}

	if load_from_disk is None:
		raise ImportError("'datasets' package is not installed. Please run: python3 -m pip install --user datasets")

	# Load local GSM8K dataset from disk: expect HuggingFace saved dataset at '<dest_root>/gsm8k/main'
	main_path = os.path.join(dest_root, 'gsm8k', 'main')
	if not os.path.exists(os.path.join(main_path, 'dataset_dict.json')):
		raise FileNotFoundError(f"Local GSM8K dataset not found at {main_path}. Expected a HuggingFace saved dataset.")
	ds = load_from_disk(main_path)
	train_ds = ds['train'].shuffle(seed=seed)
	test_ds = ds['test']

	# Build rows
	train_rows: List[Dict[str, Any]] = []
	valid_rows: List[Dict[str, Any]] = []
	test_rows: List[Dict[str, Any]] = []

	# Validation: take first valid_size from shuffled train
	for idx, ex in enumerate(train_ds):
		qa = _split_rationale_and_answer(ex['answer'])
		row = {
			"question": ex['question'],
			"answers": [qa['final']],
			"rationale": qa['rationale'],
			"all_classes": [],
			"length": len(ex['question'])
		}
		if idx < valid_size:
			valid_rows.append(row)
		else:
			train_rows.append(row)

	for ex in test_ds:
		qa = _split_rationale_and_answer(ex['answer'])
		row = {
			"question": ex['question'],
			"answers": [qa['final']],
			"rationale": qa['rationale'],
			"all_classes": [],
			"length": len(ex['question'])
		}
		test_rows.append(row)

	_write_jsonl(train_path, train_rows)
	_write_jsonl(valid_path, valid_rows)
	_write_jsonl(test_path, test_rows)

	return {"train": train_path, "valid": valid_path, "test": test_path}



if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Prepare GSM8K dataset (load local HF dataset and split).')
	parser.add_argument('--dest_root', type=str, default='.//adaptive_kv/assets/datasets', help='Destination root directory to place gsm8k/*.jsonl')
	parser.add_argument('--valid_size', type=int, default=15, help='Number of validation samples taken from train')
	parser.add_argument('--seed', type=int, default=42, help='Shuffle seed for validation sampling')
	parser.add_argument('--rebuild', action='store_true', help='Rebuild jsonl even if they exist or lack rationale field')
	args = parser.parse_args()

	paths = prepare_gsm8k_dataset(args.dest_root, valid_size=args.valid_size, seed=args.seed, rebuild=args.rebuild)
	print(json.dumps(paths, ensure_ascii=False, indent=2))
	print(f"Saved files under: {os.path.dirname(paths['train'])}")

