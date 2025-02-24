import os
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np

from funcs import (
    monolingual_dataset,
    compute_scores,
    compute_scores_optimized,
    sort_neurons_by_score,
    save_as_pickle,
    unfreeze_pickle,
)

# params
model = 'llama3'# original llama
# model = 'llama' # <- llama learned L2.
model = 'mistral'
langs = ['ja', 'nl', 'ko', 'it']
# langs = ['ja']
score_types = ['cos_sim', 'L2_dis']
# score_types = ['cos_sim']
is_last_token_only = True
n = 1000
colors = {
    'cos_sim': 'red',
    'L2_dis': 'blue',
}

def plot_histogram(data: list, model: str, n: int,  color, bins=10):
    plt.figure(figsize=(15, 15))
    plt.hist(data, bins=bins, edgecolor=None, alpha=0.8, color=color)
    plt.xlabel(f"AP Score", fontsize=30)
    plt.ylabel("Frequency", fontsize=30)
    plt.xticks(np.arange(0.1, 1.1, 0.1), fontsize=25)
    plt.yticks(fontsize=25)
    plt.title(f"AP ({L2}-specific)", fontsize=45)
    plt.savefig(
        f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/distribution/{model}/{score_type}_{L2}_n{n}',
        bbox_inches='tight',
    )

for L2 in langs:
    for score_type in score_types:
        # final scores.
        save_path_sorted_neurons = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/{score_type}/{L2}_revised.pkl'
        sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)[:n]
        
        if is_last_token_only == True:
            lang_specific_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/ap_lang_specific/ap_scores_{L2}_last_token.pkl'
        elif not is_last_token_only:
            lang_specific_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/ap_lang_specific/ap_scores_{L2}.pkl'
        ap_lang_specific = unfreeze_pickle(lang_specific_path)

        l  = []
        for neuron in sorted_neurons:
            l.append(ap_lang_specific[neuron])

        plot_histogram(l, model, n, colors[score_type])
        mean_score = np.mean(np.array(l))
        print(f'{L2}, {score_type}, is_last_token: {is_last_token_only}, mean_score: {mean_score}')
