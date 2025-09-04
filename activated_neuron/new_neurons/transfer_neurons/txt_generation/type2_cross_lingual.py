import os
import sys
sys.path.append('activated_neuron/new_neurons/transfer_neurons')
sys.path.append('activated_neuron/new_neurons/transfer_neurons/txt_generation')
import pickle
import collections
import random
import json

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from generation_funcs import (
    polywrite_with_edit_activation,
    unfreeze_pickle,
)


model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b']
model_name_dict = {
    'meta-llama/Meta-Llama-3-8B': 'Llama3-8B', 
    'mistralai/Mistral-7B-v0.3': 'Mistral-7B',
    'CohereForAI/aya-expanse-8b': 'Aya-expanse-8B',
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
langs = ['ja', 'nl', 'ko', 'it']
langs_for_polywrite = {
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
    'it': 'ita_Latn',
    'nl': 'nld_Latn',
}
deactivation_nums = [1000, 10000, 15000, 20000, 25000, 30000]
score_type = 'cos_sim'

for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("MaLA-LM/PolyWrite", split="train")

    # ★ 変更ポイント：入力言語 L_in と deactivation 言語 L_deact を分けて全組み合わせを回す（同一は除外）
    for L_in in langs:
        # 入力言語のデータを先に絞って使い回す（元の filter ロジックは維持）
        data_in = ds.filter(lambda x: x["lang_script"] == langs_for_polywrite[L_in])

        for L_deact in langs:
            if L_deact == L_in:
                continue  # 同一言語は除外

            for intervention_num in deactivation_nums:
                # ★ deactivation 側の pickle を読む（元のパス形式は維持しつつ L2 -> L_deact）
                intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L_deact}_sorted_neurons.pkl"
                intervened_neurons = unfreeze_pickle(intervened_neurons_path)

                # 元コードの絞り込みロジックはそのまま踏襲
                sorted_neurons = [neuron for neuron in intervened_neurons if neuron[0] in [_ for _ in range(20, 32)]]
                intervened_neurons = sorted_neurons[:intervention_num]

                # ★ polywrite は入力言語 L_in と deactivation のニューロンを渡す（引数順は元のまま）
                results = polywrite_with_edit_activation(
                    model, tokenizer, device, data_in, L_in, intervened_neurons, num_samples=50
                )

                # 保存。ファイル名に from(入力) と to(deactivation) を明記
                model_name_for_saving = model_name_dict[model_name]
                path = (
                    f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/txt_generation/results/type2/'
                    f'{model_name_for_saving}_{L_in}_deact-{L_deact}_intervention{intervention_num}.json'
                )
                with open(path, 'w', encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

    del model
    torch.cuda.empty_cache()