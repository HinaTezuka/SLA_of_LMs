"""
detect language specific neurons.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

import torch

from funcs import (
    compute_ap_and_sort,
    compute_ap_and_sort_np,
    save_as_pickle,
    unfreeze_pickle,
    save_np_arrays,
    unfreeze_np_arrays,
)

is_last_token_onlys = [True, False]
is_last_token_onlys = [True]
langs = ["ja", "nl", "ko", "it"]

for L2 in langs:
    """
    activations for L2.
    activations array: np.array(num_layers * num_neurons * len(data))
    """
    for is_last_token_only in is_last_token_onlys:
        # unfreeze activations.
        if not is_last_token_only:
            path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/activations/{L2}.npz"
            path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/labels/{L2}.pkl"
        elif is_last_token_only:
            path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/activations/{L2}_last_token.npz"
            path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/labels/{L2}_last_token.pkl"
        activations = unfreeze_np_arrays(path_activations)
        labels = unfreeze_pickle(path_labels)
        
        # calc AP scores.
        sorted_neurons, ap_scores = compute_ap_and_sort_np(activations, labels)

        # save AP scores as pkl.
        if not is_last_token_only:
            save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/ap_lang_specific/sorted_neurons_{L2}.pkl"
            save_path_ap_scores = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/ap_lang_specific/ap_scores_{L2}.pkl"
        elif is_last_token_only:
            save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/ap_lang_specific/sorted_neurons_{L2}_last_token.pkl"
            save_path_ap_scores = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/ap_lang_specific/ap_scores_{L2}_last_token.pkl"
        save_as_pickle(save_path_sorted_neurons, sorted_neurons)
        save_as_pickle(save_path_ap_scores, ap_scores)
        
        print(L2, "\n")
        for neuron in sorted_neurons[:10]:
            print(ap_scores[neuron])