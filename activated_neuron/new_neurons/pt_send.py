import os
import sys
sys.path.append('/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons')
import pickle
import collections
import random

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import unfreeze_pickle

# load models.
model_names = ['mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b']
model_names = ['meta-llama/Meta-Llama-3-8B', 'CohereForAI/aya-expanse-8b']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
langs = ['ja', 'nl', 'ko', 'it']
langs = ['ja', 'nl', 'ko']

# p = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/cos_sim/ja_mono_train.pkl"
# print(unfreeze_pickle(p))
# sys.exit()

score_type = 'cos_sim'
intervention_num = 1000
types = ['type_1', 'type_2']
results = {}
resutls_intervention = {}
resutls_intervention_baseline = {}
for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    for neuron_type in types:
        for L2 in langs: # L2 = deact_lang.
            """ intervention """
            if L2 == 'en':
                continue
            # intervention
            if neuron_type == 'type_1':
                # type-1
                intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
            else:
                # type-2
                intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
            intervened_neurons = unfreeze_pickle(intervened_neurons_path)
            sorted_neurons = [neuron for neuron in intervened_neurons if neuron[0] in [ _ for _ in range(20)]] if neuron_type == 'type_1' else [neuron for neuron in intervened_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
            sorted_neurons = sorted_neurons[:1000]


            # ===== npz として保存 =====
            # 形式: sorted_neurons = [(layer_idx, neuron_idx), ...]（固定長2要素タプル）
            np_arr = np.array(sorted_neurons, dtype=np.int32)
            out_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/neurons/"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{model_type}_{neuron_type}_{L2}.npz")
            np.savez_compressed(out_path, data=np_arr)
            print(f"Saved: {out_path}")
