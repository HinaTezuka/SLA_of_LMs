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

""" モジュール単位 """
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


""" W_dのvalue vector単位 """
changed_neurons_per_layer = defaultdict(list)  # {layer_idx: [neuron_indices]}

for name_a, param_a in model_ft.named_parameters():
    if "mlp.dense_4h_to_h.weight" not in name_a:
        continue  # 対象のみ絞る

    # 対応する元モデルのパラメータ取得
    param_b = dict(model_r.named_parameters())[name_a]

    # 型とデバイスを一致させる
    param_b = param_b.to(param_a.device).to(param_a.dtype)

    # name_a: 例 "transformer.h.0.mlp.dense_4h_to_h.weight"
    layer_idx = int(name_a.split(".")[2])  # 0, 1, ..., N

    # 重み行列の shape: [output_dim, input_dim]
    # 列ベクトル（input_dimの各ニューロン）を比較
    for i in range(param_a.shape[1]):
        col_a = param_a[:, i]
        col_b = param_b[:, i]
        if not torch.allclose(col_a, col_b, atol=1e-6):
            changed_neurons_per_layer[layer_idx].append(i)

# 出力結果
c = 0
for layer, changed_indices in changed_neurons_per_layer.items():
    c += len(changed_indices)
    print(f"Layer {layer}: {len(changed_indices)} neurons changed → {changed_indices[:10]}{' ...' if len(changed_indices) > 10 else ''}")
print(c)