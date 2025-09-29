## Repository Structure

```
├── adaptive_kv/                    # Core adaptive KV implementation
│   ├── assets/                     # Pre-computed scores and datasets
│   │   ├── datasets/              # Dataset files
│   │   └── head_scores/           # Pre-computed attention head scores
│   └── monkeypatch/               # Model-specific implementations
│       ├── adaptive_qwen3_hijack.py   # Qwen3 model adaptations
│       ├── monkeypatch.py             # Main monkey-patching logic
│       └── utils.py                   # Utility functions
├── experiments/                    # Experimental scripts and evaluations
│   ├── gsm8k/                     # GSM8K math reasoning experiments
│   ├── longbench/                 # LongBench evaluation scripts
│   ├── math/                      # Mathematical reasoning tasks
│   ├── memory_latency/            # Memory and latency benchmarks
│   └── needle/                    # Needle-in-haystack experiments
└── environment.yml                # Conda environment specification
```
code_llama_mistral                 #Code of Llama & Mistrial

## Installation

### Setup Environment



1. **Create conda environment:**
```bash
conda env create -f environment.yml
conda activate qwen3
```

1. **Install additional dependencies (if needed):**
```bash
pip install transformers torch numpy
```

### Model Setup

Update model paths in the configuration files and scripts:
- Modify `/path/to/models/` paths in experiment scripts to point to your model directory
- Ensure you have access to the required models (Qwen3-32B, LLaMA, Mistral variants)

## Usage


### Running Experiments
#### LongBench Evaluation

```bash
cd experiments/longbench
python qwen3_inference.py \
    --model_name_or_path /path/to/models/Qwen3-32B \
    --max_length 32768 \
    --compress_args_path c128_w32_k7_maxpool.json \
    --out_name qwen3_cokv_results
```

#### GSM8K Math Reasoning

```bash
cd experiments/gsm8k
python inference.py \
    --model_name_or_path /path/to/models/Qwen3-32B \
    --mode sv \
    --compress_args_path c128_w32_k7_maxpool.json
```

#### Memory and Latency Analysis

```bash
cd experiments/memory_latency
python memory.py  # For memory usage analysis
python latency.py # For latency benchmarking
```

### Configuration Files

Configuration files in `experiments/longbench/config/` control cache behavior:

- `c64_w32_k7_maxpool.json`: 64 cache size configuration
- `c128_w32_k7_maxpool.json`: 128 cache size configuration  
- `c256_w32_k7_maxpool.json`: 256 cache size configuration
- `c512_w32_k7_maxpool.json`: 512 cache size configuration
- `c1024_w32_k7_maxpool.json`: 1024 cache size configuration


