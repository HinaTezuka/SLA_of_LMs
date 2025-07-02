import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import defaultdict

from funcs import (
    save_as_pickle,
    unfreeze_pickle,
)

model_types = ['aya', 'llama3', 'mistral']
# model_types = ['mistral']
langs = ['ja', 'ko', 'fr']
num_to_extract = 20
L2 = 'fr'
model_type = 'aya'

# 入力ファイル（.jsonl）
input_path = "/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/aya_fr_normal_20.jsonl/CohereForAI__aya-expanse-8b/samples_mmlu_prox_fr_history_2025-07-01T10-00-04.953122.jsonl"

# 出力ファイル（.json）
output_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/output_samples/{model_type}_normal_{L2}_{num_to_extract}.json"

# resps を格納するリスト
resps_list = []

# 読み込み処理
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        try:
            resps = data["resps"]
            resps_list.append(resps)
        except KeyError:
            continue  # resps がない行はスキップ

# 保存（可視的な JSON 形式で）
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(resps_list, f, ensure_ascii=False, indent=2)

print(f"{len(resps_list)} 件の出力を '{output_path}' に保存しました。")