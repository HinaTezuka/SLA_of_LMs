import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from mknn_funcs import (
    compute_mutual_knn,
    compute_mutual_knn_with_edit_activation,
)
from funcs import (
    save_as_pickle,
    unfreeze_pickle,
    unfreeze_np_arrays,
)

model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
langs = ['ja', 'nl', 'ko', 'it']
L1 = 'en'
topk = 10 # number of nearest neighbor.
# is_reverses = [False, True]
is_reverses = [False]
score_type = 'cos_sim'
intervention_num = 1000

for model_name in model_names:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    knn_scores = {}
    for is_reverse in is_reverses:
        for L2 in langs:
            path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_multi_train.pkl'
            sentences = unfreeze_pickle(path)

            if not is_reverse:
                path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
                neurons = unfreeze_pickle(path)
                neurons = [neuron for neuron in neurons if neuron[0] in [ _ for _ in range(20)]]
            elif is_reverse:
                path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
                neurons = unfreeze_pickle(path)
                neurons = [neuron for neuron in neurons if neuron[0] in [ _ for _ in range(20, 32)]]
            neurons = neurons[:intervention_num]

            res = compute_mutual_knn_with_edit_activation(model, tokenizer, device, sentences, L1, L2, topk, neurons) # res: [knn_score_layer1, knn_score_layer2, ...]
            print(f'=================={model_type}, {L2}==================')
            # print(res)
            knn_scores[L2] = res
    
        if not is_reverse:
            path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/knn/res_all_langs_k{topk}_type1_n{intervention_num}.pkl'
        elif is_reverse:
            path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/knn/res_all_langs_k{topk}_type2_n{intervention_num}.pkl'
        save_as_pickle(path, knn_scores)

    # clear cache.
    del model
    torch.cuda.empty_cache()

""" visualization. """
model_types = ['llama3', 'mistral', 'aya']
languages = ['ja', 'nl', 'ko', 'it']
# Prepare a list to collect all rows for the DataFrame
all_data = []

for is_reverse in is_reverses:
    # Load and reshape data for each model
    for model_type in model_types:
        if not is_reverse:
            path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/knn/res_all_langs_k{topk}_type1_n{intervention_num}.pkl'
        elif is_reverse:
            path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/knn/res_all_langs_k{topk}_type2_n{intervention_num}.pkl'
        model_data = unfreeze_pickle(path)
        
        for lang in languages:
            values = model_data[lang]  # A list of length equal to the number of layers
            for layer, value in enumerate(values):
                all_data.append({
                    'Model': model_type,
                    'Layer': layer,
                    'Mutual KNN': value,
                    'L2': lang
                })

    # Convert to a pandas DataFrame
    df = pd.DataFrame(all_data)

    # Generate a separate plot for each model
    for model_type in model_types:
        plt.figure(figsize=(10, 6))
        subset = df[df['Model'] == model_type]
        sns.lineplot(data=subset, x='Layer', y='Mutual KNN', hue='L2', palette='tab10', linewidth=3)
        model_name = 'LLaMA3-8B' if model_type == 'llama3' else 'Mistral-7B' if model_type == 'mistral' else 'Aya-expanse-8B'
        plt.title(f'{model_name}', fontsize=30)
        plt.xlabel('Layer Index', fontsize=35)
        plt.ylabel('Mutual KNN', fontsize=35)
        plt.ylim(0, 0.6)
        plt.tick_params(axis='both', labelsize=15)
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title='L2', fontsize=15, title_fontsize=15)
        
        if not is_reverse:
            save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/mutual_knn/{model_type}_top{topk}_type1_n{intervention_num}.png'
        elif is_reverse:
            save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/mutual_knn/{model_type}_top{topk}_type2_n{intervention_num}.png'
        plt.savefig(save_path, bbox_inches='tight')