import os
import sys
import copy
import pickle
from collections import defaultdict

import numpy as np

from funcs import (
    save_as_pickle,
    unfreeze_pickle,
    save_np_arrays,
    unfreeze_np_arrays,
)

langs = ['ja', 'nl', 'ko', 'it']
model_types = ['llama3', 'mistral', 'aya']
score_types = ['cos_sim', 'L2_dis']

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
            # activations
            save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/activations/{L2}_last_token.npz"
            save_path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/labels/{L2}_last_token.pkl"
            activations_arr = unfreeze_np_arrays(save_path_activations)
            labels_list = np.array(unfreeze_pickle(save_path_labels))

            corr_ratios = np.zeros((layers_num, neurons_num))
            arr = []
            for layer_i in range(layers_num):
                for neuron_i in range(neurons_num):
                    corr_ratio = correlationRatio(labels_list, activations_arr[layer_i, neuron_i, :])
                    corr_ratios[layer_i, neuron_i] = corr_ratio
            
            path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/corr_ratio/cos_sim/{model_type}_{L2}'
            save_np_arrays(path, corr_ratios)

            print(f'{L2}, {score_type}, {model_type}, completed.')
            # print(np.mean(np.array(arr)))