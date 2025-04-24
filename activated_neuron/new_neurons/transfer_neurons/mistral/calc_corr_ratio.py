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
model = 'mistral'
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
        save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mistral/activations/{L2}_last_token.npz"
        save_path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mistral/labels/{L2}_last_token.pkl"
        activations_arr = unfreeze_np_arrays(save_path_activations)
        labels_list = np.array(unfreeze_pickle(save_path_labels))
        # top score neurons
        save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mistral/final_scores/{score_type}/{L2}_mono_train.pkl"
        # save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mistral/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
        sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
        # sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
        sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20)]]
        """ test for langage families. """
        # l1 = [ 1 for _ in range(1000)]
        # l2 = [ 0 for _ in range(1000)]
        # if L2 in ['nl', 'it']:
        #     labels_list = l2 + l1 + copy.deepcopy(l2) + copy.deepcopy(l1) + copy.deepcopy(l1) # nl, it
        # elif L2 in ['ja', 'ko']:
        #     labels_list = l1 + copy.deepcopy(l2) + copy.deepcopy(l1) + copy.deepcopy(l2) + copy.deepcopy(l2) # ja, koo

        top_n = 1000
        corr_ratios = defaultdict(float)
        arr = []
        for layer_i, neuron_i in sorted_neurons[:top_n]:
            corr_ratio = correlationRatio(labels_list, activations_arr[layer_i, neuron_i, :])
            corr_ratios[(layer_i, neuron_i)] = corr_ratio
            arr.append(corr_ratio)

        print(f'{L2}, {score_type}')
        print(np.mean(np.array(arr)))