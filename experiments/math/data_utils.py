import os
import json
from typing import Dict, Any, List

try:
    from datasets import load_from_disk
except Exception:
    load_from_disk = None


def _extract_math_final(solution: str) -> str:
    """Extract a concise final answer token from MATH/competition_math solution text.

    Heuristics:
    - Prefer the last \boxed{...} content if present.
    - Otherwise, take the last number-like token (int/float/fraction/scientific).
    - Normalize by stripping commas and trailing period.
    """
    if not solution:
        return ""
    s = solution.strip()

    # Prefer last \boxed{...}
    import re
    boxed = list(re.finditer(r"\\boxed\{([^}]*)\}", s))
    token = None
    if boxed:
        token = boxed[-1].group(1)
    else:
        # Try common patterns like "Answer:" or similar
        tail = s.split("Answer:")[-1] if "Answer:" in s else s
        # Last numeric-like token (supports fractions)
        m = re.findall(r"[+-]?\d*[\.,]?\d+(?:[eE][+-]?\d+)?|[+-]?\d+\s*/\s*\d+", tail)
        if m:
            token = m[-1]
    if token is None:
        return ""
    token = token.replace(",", "").strip()
    if token.endswith('.'):
        token = token[:-1]
    return token.strip()


def _write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def prepare_math_dataset(dest_root: str, valid_size: int = 500, test_size: int = 1000, seed: int = 42, rebuild: bool = False) -> Dict[str, str]:
    """Build train/valid/test jsonl for MATH (competition_math) from local HF dataset.

    Expected HF dataset path: <dest_root>/math (with only 'train' split present). We'll split it.
    Output jsonl files will include fields: question, answers [final], rationale, all_classes, length.
    """
    out_dir = os.path.join(dest_root, 'math')
    train_path = os.path.join(out_dir, 'train.jsonl')
    valid_path = os.path.join(out_dir, 'valid.jsonl')
    test_path = os.path.join(out_dir, 'test.jsonl')

    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path) and not rebuild:
        return {"train": train_path, "valid": valid_path, "test": test_path}

    if load_from_disk is None:
        raise ImportError("'datasets' package is not installed. Please run: python3 -m pip install --user datasets")

    # Load local MATH dataset saved at '<dest_root>/math'
    main_path = os.path.join(dest_root, 'math')
    if not os.path.exists(os.path.join(main_path, 'dataset_dict.json')):
        raise FileNotFoundError(f"Local MATH dataset not found at {main_path}. Expected a HuggingFace saved dataset.")
    ds = load_from_disk(main_path)
    train_ds = ds['train'].shuffle(seed=seed)

    n = len(train_ds)
    v = max(0, min(valid_size, n))
    t = max(0, min(test_size, n - v))

    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for idx, ex in enumerate(train_ds):
        q = ex.get('problem', '')
        sol = ex.get('solution', '')
        final = _extract_math_final(sol)
        row = {
            "question": q,
            "answers": [final],
            "rationale": sol,
            "all_classes": [ex.get('level', ''), ex.get('type', '')],
            "length": len(q)
        }
        if idx < v:
            valid_rows.append(row)
        elif idx < v + t:
            test_rows.append(row)
        else:
            train_rows.append(row)

    # If no explicit test requested, carve a small test from tail
    if not test_rows and valid_rows:
        # move last min(1000, len(valid_rows)//2) from valid to test
        k = min(1000, len(valid_rows) // 2)
        test_rows = valid_rows[-k:]
        valid_rows = valid_rows[:-k]

    _write_jsonl(train_path, train_rows)
    _write_jsonl(valid_path, valid_rows)
    _write_jsonl(test_path, test_rows)

    return {"train": train_path, "valid": valid_path, "test": test_path}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare MATH (competition_math) dataset: split and export jsonl')
    parser.add_argument('--dest_root', type=str, default='.//adaptive_kv/assets/datasets', help='Destination root directory to place math/*.jsonl')
    parser.add_argument('--valid_size', type=int, default=500, help='Validation split size')
    parser.add_argument('--test_size', type=int, default=1000, help='Test split size')
    parser.add_argument('--seed', type=int, default=42, help='Shuffle seed')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild even if files exist')
    args = parser.parse_args()

    paths = prepare_math_dataset(args.dest_root, valid_size=args.valid_size, test_size=args.test_size, seed=args.seed, rebuild=args.rebuild)
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    print(f"Saved files under: {os.path.dirname(paths['train'])}")
