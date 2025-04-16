"""
compare acc of QA for both normal and deactivated model.
"""
import os
import sys
import pickle
import collections
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

from qa_funcs import (
    mkqa_for_scatter_plot_THRESHOLD,
    mkqa_for_scatter_plot_with_edit_activation,
    save_as_pickle,
    unfreeze_pickle,
)

# load models (LLaMA3-8B).
model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
langs = ['ja', 'nl', 'ko', 'it']
""" 
QA dataset: 
MKQA: Multilingual Open Domain Question Answering
・https://arxiv.org/abs/2007.15207
・https://github.com/apple/ml-mkqa/
・https://huggingface.co/datasets/apple/mkqa
"""
# qa_num = 100
qa = load_dataset('apple/mkqa')['train']
# qa = qa.shuffle(seed=42)
score_type = 'cos_sim'
intervention_num = 1000

results = {}
resutls_intervention = {}
resutls_intervention_baseline = {}
for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for L2 in langs:
        # load question indices.
        THRESHOLD = 0.5
        path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qa/{model_type}_qa_indices_above_{THRESHOLD}_all_langs.pkl'
        qa_indices_dict = unfreeze_pickle(path)
        # normal
        result_scores = mkqa_for_scatter_plot_THRESHOLD(model, tokenizer, device, qa, L2, qa_indices_dict)
        results[L2] = result_scores

        """ intervention """
        if L2 == 'en':
            continue
        # intervention
        intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
        intervened_neurons = unfreeze_pickle(intervened_neurons_path)
        intervened_neurons_main = intervened_neurons[:intervention_num]
        result_score = mkqa_for_scatter_plot_with_edit_activation(model, tokenizer, device, qa, L2, intervened_neurons_main, qa_indices_dict)
        resutls_intervention[L2] = result_score
        # intervention baseline.
        random.seed(42)
        intervened_neurons_baseline = random.sample(intervened_neurons[intervention_num+1:], len(intervened_neurons[intervention_num+1:]))
        intervened_neurons_baseline = intervened_neurons_baseline[:intervention_num]
        result_score = mkqa_for_scatter_plot_with_edit_activation(model, tokenizer, device, qa, L2, intervened_neurons_baseline, qa_indices_dict)
        resutls_intervention_baseline[L2] = result_score

    """ visualization. """
    def plot_intervention_scatter(dict_normal, dict_intervene1, dict_intervene2):
        """
        For each language, plots a scatter plot:
        X-axis: f1_score from dict_normal(no intervention).
        Y-axis: f1_score from dict_intervene1(top1000 intervention) and dict_intervene2(intervention_baseline).
        
        Each language gets its own subplot.
        Interventions are shown with different colors and markers.
        """
        dicts = [dict_intervene1, dict_intervene2]
        dict_labels = ['TransferNeurons', 'Baseline']
        markers = ['s', '^']
        colors = ['green', 'red']

        # Union of all languages
        all_languages = sorted(set(dict_normal.keys()) | set(dict_intervene1.keys()) | set(dict_intervene2.keys()))
        
        n_langs = len(all_languages)
        n_cols = 2
        n_rows = (n_langs + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

        for idx, lang in enumerate(all_languages):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]

            base_data = dict_normal.get(lang, [])
            base_f1s = [f1 for _, f1 in base_data]
            
            for d_idx, d in enumerate(dicts):
                intervened_data = d.get(lang, [])
                if not intervened_data or not base_data or len(intervened_data) != len(base_data):
                    continue
                intervened_f1s = [f1 for _, f1 in intervened_data]
                ax.scatter(
                    base_f1s,
                    intervened_f1s,
                    label=dict_labels[d_idx],
                    marker=markers[d_idx],
                    color=colors[d_idx],
                    alpha=0.7
                )

            # Plot y=x line for reference
            min_f1 = min(base_f1s + [f for d in dicts for (_, f) in d.get(lang, [])])
            max_f1 = max(base_f1s + [f for d in dicts for (_, f) in d.get(lang, [])])
            ax.plot([min_f1, max_f1], [min_f1, max_f1], linestyle='--', color='gray', linewidth=1)

            ax.set_title(f'F1 Score: {lang}', fontsize=30)
            if idx == 0:
                ax.set_xlabel('Normal', fontsize=20)
                ax.set_ylabel('Intervened', fontsize=20)
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.tick_params(axis='both', labelsize=15)
            ax.legend()
            ax.grid(True)

        # Remove any unused subplots
        for i in range(n_langs, n_rows * n_cols):
            fig.delaxes(axes[i // n_cols][i % n_cols])
        
        plt.tight_layout()
        path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/qa/scatter_f1_{model_type}.png'
        plt.savefig(
            path,
            bbox_inches='tight'
        )
    
    plot_intervention_scatter(results, resutls_intervention, resutls_intervention_baseline)
    print(f'plot suceeded: {model_type}')

    del model
    torch.cuda.empty_cache()