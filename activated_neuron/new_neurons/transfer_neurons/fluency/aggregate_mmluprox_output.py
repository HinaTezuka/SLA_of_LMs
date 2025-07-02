import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import defaultdict

from funcs import (
    save_as_pickle,
    unfreeze_pickle,
)
# p = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/aya/ja_baseline.pkl'
# l = unfreeze_pickle(p)
# print(l)
# sys.exit()

def extract_resps_from_jsonl(filepath, n):
    resps_list = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if len(resps_list) >= n:
                break
            try:
                data = json.loads(line)
                resps = data.get("resps", [])
                if isinstance(resps, list):
                    for resp in resps:
                        if isinstance(resp, list):
                            resps_list.extend(resp)
                        elif isinstance(resp, str):
                            resps_list.append(resp)
                else:
                    continue
            except json.JSONDecodeError:
                print("Skipping invalid JSON line.")
                continue

    return resps_list[:n]

import json

def convert_pickle_top_n_to_json(pickle_path, json_output_path, n=20):
    # pickle を読み込み
    data = unfreeze_pickle(pickle_path)

    # 先頭 n 件だけ抽出
    top_n_data = data[:n]

    # JSON に保存（人間可読）
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(top_n_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 上位 {n} 件のレスポンスを '{json_output_path}' に保存しました。")

model_types = ['aya', 'llama3']
# model_types = ['mistral']
langs = ['ja', 'ko', 'fr']
# is_baselines = [False, True]
is_baselines = [False]
num_to_extract = 100

for model_type in model_types:
    for L2 in langs:
        for is_baseline in is_baselines:
            # if is_baseline:
            #     file_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/output_samples/{model_type}_{L2}_baseline.jsonl'
            # else: # type-1 deactivated.
            #     file_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/output_samples/{model_type}_{L2}_type1.jsonl'
            
            # resps_output = extract_resps_from_jsonl(file_path, num_to_extract)

            save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{L2}.pkl' if not is_baseline else f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{L2}_baseline.pkl'
            # save_as_pickle(save_path, resps_output)

            json_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/output_samples/type1_deactivation_{model_type}_{L2}.json'
            convert_pickle_top_n_to_json(save_path, json_path, n=20)