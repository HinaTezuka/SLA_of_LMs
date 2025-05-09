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
model_names = ['meta-llama/Meta-Llama-3-8B']
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
THRESHOLD = 0

for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'

    if model_type == 'llama3':
        normal_dict_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/qa/all_questions.pkl'
        normal_dict = unfreeze_pickle(normal_dict_path)

    def get_q_above_th(THRESHOLD: int, normal: list, intervention: list, intervention_baseline: list):
        normal_r = []
        intervention_r = []
        intervention_baseline_r = []

        def get_q_idx_and_f1(score_list, target_idx):
            for q_idx, f1_score in score_list:
                if q_idx == target_idx:
                    return q_idx, f1_score
        
        normal_scores = []
        intervention_scores = []
        intervention_baseline_scores = []
        for q_idx, f1_score in normal:
            if f1_score >= THRESHOLD:
                n_score = f1_score
                normal_r.append((q_idx, f1_score))
                normal_scores.append(n_score)
                i_idx, i_score = get_q_idx_and_f1(intervention, q_idx) # i_idx: q_idx(intervention), i_score: f1_score(intervention)
                intervention_r.append((i_idx, i_score))
                intervention_scores.append(i_score)
                b_idx, b_score = get_q_idx_and_f1(intervention_baseline, q_idx) # b_idx: q_idx(intervention_baseline), b_score: f1_score(intervention_baseline)
                intervention_baseline_r.append((b_idx, b_score))
                intervention_baseline_scores.append(b_score)
        
        return normal_r, intervention_r, intervention_baseline_r, np.mean(np.array(normal_scores)), np.mean(np.array(intervention_scores)), np.mean(np.array(intervention_baseline_scores))

    normal_dict = {}
    intervention_dict = {}
    intervention_baseline_dict = {}

    if model_type == 'llama3':
        normal_dict_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/qa/all_questions.pkl'
        normal_llama = unfreeze_pickle(normal_dict_path)
        
    for L2 in langs:
        if model_type == 'llama3':
            normal = normal_llama[L2]
        else:
            normal_list_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/all_questions_normal_{L2}.pkl'
            normal = unfreeze_pickle(normal_list_path)
        intervention_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/intervention_n1000/all_questions_intervention_{L2}.pkl'
        intervention_baseline_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/intervention_n1000/all_questions_baseline_{L2}.pkl'
        intervention = unfreeze_pickle(intervention_path)
        intervention_baseline = unfreeze_pickle(intervention_baseline_path)

        # get q_idx and f1 above THRESHOLD.
        normal_l, intervention_l, intervention_baseline_l, normal_mean_score, intervention_mean_score, intervention_baseline_mean_score = get_q_above_th(THRESHOLD, normal, intervention, intervention_baseline)
        normal_dict[L2] = normal_l
        intervention_dict[L2] = intervention_l
        intervention_baseline_dict[L2] = intervention_baseline_l
        print(f'{model_type}, {L2}: normal:{normal_mean_score}, intervention:{intervention_mean_score}, intervention_baseline:{intervention_baseline_mean_score}')

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
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

        for idx, lang in enumerate(all_languages):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            ax.set_facecolor('lightgray') # background color.

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
                    alpha=0.8,
                )

            # Plot y=x line for reference
            min_f1 = min(base_f1s + [f for d in dicts for (_, f) in d.get(lang, [])])
            max_f1 = max(base_f1s + [f for d in dicts for (_, f) in d.get(lang, [])])
            ax.plot([min_f1, max_f1], [min_f1, max_f1], linestyle='--', color='blue', linewidth=2)

            ax.set_title(f'{lang}', fontsize=30)
            if idx == 0:
                ax.set_xlabel('Normal', fontsize=20)
                ax.set_ylabel('Intervened', fontsize=20)
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.tick_params(axis='both', labelsize=10)
            ax.legend()
            ax.grid(True)

        # Remove any unused subplots
        for i in range(n_langs, n_rows * n_cols):
            fig.delaxes(axes[i // n_cols][i % n_cols])
        
        plt.tight_layout()
        path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/qa/{model_type}_above{THRESHOLD}.png' if THRESHOLD != 0 else f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/qa/{model_type}_all.png'
        plt.savefig(
            path,
            bbox_inches='tight'
        )
    plot_intervention_scatter(normal_dict, intervention_dict, intervention_baseline_dict)
    print(f'plot suceeded: {model_type}')

    torch.cuda.empty_cache()