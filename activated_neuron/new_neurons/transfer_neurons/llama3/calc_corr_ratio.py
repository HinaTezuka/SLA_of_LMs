import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import pickle
from collections import defaultdict

import numpy as np

from funcs import (
    multilingual_dataset_for_lang_specific_detection,
    track_neurons_with_text_data,
    save_as_pickle,
    compute_ap_and_sort,
    unfreeze_pickle,
    save_np_arrays,
    unfreeze_np_arrays,
)

langs = ['ja', 'nl', 'ko', 'it']
# langs = ['nl', 'it']
# langs = ['ja', 'ko']
model = 'llama3'
score_types = ['cos_sim', 'L2_dis']

def correlationRatio(categories, values):
    interclass_variation  = sum([
        (len(values[categories == i]) * ((values[categories == i].mean() - values.mean()) ** 2)).sum() for i in np.unique(categories)
    ]) 
    total_variation = sum((values - values.mean()) ** 2)
    return interclass_variation / total_variation

for L2 in langs:
    for score_type in score_types:
        # activations
        save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/activations/{L2}_last_token.npz"
        save_path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/labels/{L2}_last_token.pkl"
        activations_arr = unfreeze_np_arrays(save_path_activations)
        labels_list = np.array(unfreeze_pickle(save_path_labels))
        # l1 = [ 1 for _ in range(1000)]
        # l2 = [ 0 for _ in range(1000)]
        # labels_list = l2 + l1 + copy.deepcopy(l2) + copy.deepcopy(l1) + copy.deepcopy(l1) # nl, it
        # labels_list = l1 + copy.deepcopy(l2) + copy.deepcopy(l1) + copy.deepcopy(l2) + copy.deepcopy(l2) # ja, ko
        # top score neurons
        save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/{score_type}/{L2}_mono_train.pkl"
        # save_path_sorted_neurons = '/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/sorted_neurons_ja_last_token.pkl'
        sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)

        top_n = 5000
        corr_ratios = defaultdict(float)
        arr = []
        for (layer_i, neuron_i) in sorted_neurons[:top_n]:
            corr_ratio = correlationRatio(labels_list, activations_arr[layer_i, neuron_i, :])
            corr_ratios[(layer_i, neuron_i)] = corr_ratio
            arr.append(corr_ratio)

        print(f'{L2}, {score_type}')
        print(np.mean(np.array(arr)))