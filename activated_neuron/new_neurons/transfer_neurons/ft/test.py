import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import defaultdict

import numpy as np
import torch
import transformers
import wandb
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_ft = AutoModelForCausalLM.from_pretrained('HinataTezuka/FT-TN-ja-bloom-1000').to(device)
model_r = AutoModelForCausalLM.from_pretrained('bigscience/bloom-3b', torch_dtype=torch.bfloat16).to(device)

# モジュール単位で bool を保持
module_results = defaultdict(list)

for (name_a, param_a), (name_b, param_b) in zip(model_ft.named_parameters(), model_r.named_parameters()):
    if name_a != name_b:
        print(f"WARNING: Name mismatch → {name_a} != {name_b}")
    
    # デバイス & dtype 揃える
    param_b = param_b.to(param_a.device).to(param_a.dtype)
    
    # 比較
    is_equal = torch.allclose(param_a, param_b, atol=1e-6)
    
    # モジュール名（例： "transformer.h.0.self_attention"）
    module_name = ".".join(name_a.split(".")[:-1])
    
    # 結果を append
    module_results[module_name].append(is_equal)

# モジュールごとの結果（全部 True → True, 1つでも False → False）
for module, results in module_results.items():
    overall_result = all(results)
    print(f"{module}: {overall_result}")