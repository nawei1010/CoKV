# CoKV

## Usage

### Requirements
please see the **environment.yml**
(replace transformers==4.40.0 when you use mistral-7b-v0.2 model), see /CoKV/csrc/makefile to install tiny-pkg

please set **project_path='Path_To_Your_Project/CoKV'**

please use /CoKV/assets/datasets/process_longbench_split.py to split valid and test dataset

## Evaluations
### importance score computation
For LongBench, please cd **/CokV/experiments/LongBench/importance_evaluation** and run **./cal_global_sv.sh**.

For NIAH, please cd **/CokV/experiments/Needle/importance_evaluation** and run **./needle_in_haystack.sh**.

We recommend performing two independent sampling runs to get stable Sliced Shapley value. The results are considered stable when the mean absolute error between the Sliced Shapley values from the two runs is less than  1/n where n is your player. The --sampling_number 10000 parameter is intended to keep the script running indefinitely, not to enforce exactly 10,000 samplings. You can adjust the script to terminate once the error falls below your specified threshold.

### experiment in the paper
For benchmark results , see /CoKV/experiments/LongBench/model_inference/pred.py and pred.sh

For mask head experiments, see /CoKV/experiments/LongBench/model_inference/delete_head.py and delete_head.sh

For cross dataset mask head experiments, see /CoKV/experiments/LongBench/model_inference/cross_head_mask.py and cross_head_mask.sh

For longbench results, run eval.py to get test accuracy

For NIAH test, see /CoKV/experiments/Needle/model_inference/needle_in_haystack.py and needle_in_haystack.sh

If you want use the provided Sliced Shapley values, you can find them at /CoKV/adaptive_kv/cc_scores.

### Quick Start

```python
# replace modeling with adaptive kv cache when using 'ada','snapkv','headkv','cokv'
```
from adaptive_kv.monkeypatch.monkeypatch import replace_mistral_adaptive, replace_llama_adaptive
replace_mistral_adaptive()
replace_llama_adaptive()

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    config=config,
    device_map=device_map,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

when you choose mode in ['snapkv','headkv','cokv'], the parameter model.model.config.given_adaptive_size will be set.
when you choose mode='ada', model.model.config.given_adaptive_size will be none.
### Acknowledgments​
We gratefully acknowledge the following open-source projects that contributed to our implementation:

- [​SnapKV](https://github.com/FasterDecoding/SnapKV)​
- [AdaKV](https://github.com/FFY0/AdaKV)
- [HeadKV](https://github.com/FYYFU/HeadKV)​
- [LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack?tab=readme-ov-file)
  
We appreciate the researchers and developers for making their code publicly available, which significantly advanced our work.




