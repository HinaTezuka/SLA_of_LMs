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
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

from generation_funcs import (
    polywrite_with_edit_activation,
    unfreeze_pickle,
)

# load models (LLaMA3-8B).
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
is_baselines = [False, True]

for is_baseline in is_baselines:
    for model_name in model_names:
        model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ds = load_dataset("MaLA-LM/PolyWrite", split="train")

        for L2 in langs:
            for intervention_num in deactivation_nums:
                intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
                intervened_neurons = unfreeze_pickle(intervened_neurons_path)
                sorted_neurons = [neuron for neuron in intervened_neurons if neuron[0] in [ _ for _ in range(20)]]
                intervened_neurons = sorted_neurons[:intervention_num]

                if is_baseline:
                    # intervention baseline.
                    random.seed(42)
                    intervened_neurons_baseline = random.sample(sorted_neurons[intervention_num:], intervention_num)
                    intervened_neurons= intervened_neurons_baseline[:intervention_num]

                data = ds.filter(lambda x: x["lang_script"]==langs_for_polywrite[L2])
                results = polywrite_with_edit_activation(model, tokenizer, device, data, L2, intervened_neurons, num_samples=50)
            
                # save.
                model_name_for_saving = model_name_dict[model_name]
                path = f'activated_neuron/new_neurons/transfer_neurons/txt_generation/results/type1/{model_name_for_saving}_{L2}_intervention{intervention_num}.json' if not is_baseline else f'activated_neuron/new_neurons/transfer_neurons/txt_generation/results/type1/{model_name_for_saving}_{L2}_intervention{intervention_num}_baseline.json'
                with open(path, 'w') as f:
                    json.dump(results, f, indent=4)