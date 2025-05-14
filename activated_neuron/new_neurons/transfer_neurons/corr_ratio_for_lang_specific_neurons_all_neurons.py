"""
corr ratio for language-specificity.
"""
import os
import sys
import copy
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import average_precision_score

from funcs import (
    save_as_pickle,
    unfreeze_pickle,
    save_np_arrays,
    unfreeze_np_arrays,
)

langs = ['ja', 'nl', 'ko', 'it']
model_types = ['llama3', 'mistral', 'aya']
score_types = ['cos_sim', 'L2_dis']

def plot(matrix, th, model_type, L2, score_type):
    """
    Plot a bar chart of how many (layer, neuron) positions exceed a threshold for each layer.

    Args:
        matrix (np.ndarray): 2D array of shape (layer_num, neuron_num), containing values (e.g., activations or scores).
        th (float): Threshold. Only values > th are counted.
        model_type (str): Model type name, used in file path.
        L2 (str): Label or suffix, used in file path.
        title (str): Title of the bar plot.
    """
    # Boolean mask where values exceed the threshold
    above_th = matrix > th

    # Sum along neurons (axis=1), to get counts per layer
    layer_counts = np.sum(above_th, axis=1)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(matrix.shape[0]), layer_counts, color='steelblue')
    plt.xlabel('Layer Index')
    plt.ylabel(f'Count of neurons with value > {th}')
    plt.title(f'{model_type}: {L2}', fontsize=30)
    plt.xticks(np.arange(matrix.shape[0]))
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tight_layout()

    # Create save directory if it doesn't exist
    save_dir = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/distribution/{model_type}/lang_specific/corr_{L2}_{score_type}_{th}'
    # os.makedirs(save_dir, exist_ok=True)

    # # Save figure
    # save_path = os.path.join(save_dir, f'corr_{L2}_{score_type}_{th}.png')
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.close()
    with PdfPages(save_dir + '.pdf') as pdf:
        pdf.savefig(bbox_inches='tight', pad_inches=0.01)
        plt.close()

def correlationRatio(categories, values):
    interclass_variation  = sum([
        (len(values[categories == i]) * ((values[categories == i].mean() - values.mean()) ** 2)).sum() for i in np.unique(categories)
    ]) 
    total_variation = sum((values - values.mean()) ** 2)
    return interclass_variation / total_variation

layers_num = 32
neurons_num = 14336

for score_type in score_types:
    for model_type in model_types:
        for L2 in langs:
            # # activations
            # save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/activations/{L2}_last_token.npz"
            # save_path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/labels/{L2}_last_token.pkl"
            # activations_arr = unfreeze_np_arrays(save_path_activations)
            # labels_list = np.array(unfreeze_pickle(save_path_labels))

            # corr_ratios = np.zeros((layers_num, neurons_num))
            arr = []
            # for layer_i in range(layers_num):
            #     for neuron_i in range(neurons_num):
                    # corr_ratio = correlationRatio(labels_list, activations_arr[layer_i, neuron_i, :])
                    # corr_ratios[layer_i, neuron_i] = corr_ratio

            
            # path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/corr_ratio/{score_type}/{model_type}_{L2}'
            # save_np_arrays(path, corr_ratios)

            print(f'{L2}, {score_type}, {model_type}, completed.')

            """ visualization """
            ths = [0.1, 0.25, 0.5]
            for th in ths:
                path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/corr_ratio/{score_type}/{model_type}_{L2}.npz'
                corr_ratios = unfreeze_np_arrays(path)
                plot(corr_ratios, th, model_type, L2, score_type)