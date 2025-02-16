"""
detect language specific neurons.
"""
import sys
import pickle

import torch

from funcs import (
    compute_ap_and_sort,
    save_as_pickle,
    unfreeze_pickle,
)

# LLaMA-3
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}
# params
device = "cuda" if torch.cuda.is_available() else "cpu"

is_last_token_onlys = [False, True]

for L2, model_name in model_names.items():
    """
    activations for L2.
    activation_dict
    {
        text_idx:
            layer_idx: [(neuron_idx, act_value), (neuron_idx, act_value), ....]
    }
    """
    for is_last_token_only in is_last_token_onlys:
        # unfreeze activations.
        if not is_last_token_only:
            path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/activations/{L2}_normal.pkl"
            path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/labels/{L2}_normal.pkl"
        elif is_last_token_only:
            path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/activations/{L2}_last_token.pkl"
            path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/labels/{L2}_last_token.pkl"
        activations = unfreeze_pickle(path_activations)
        labels = unfreeze_pickle(path_labels)
        
        # calc AP scores.
        sorted_neurons, ap_scores = compute_ap_and_sort(activations, labels)

        # save AP scores as pkl.
        if not is_last_token_only:
            save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/sorted_neurons_{L2}.pkl"
            save_path_ap_scores = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/ap_scores_{L2}.pkl"
        elif is_last_token_only:
            save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/sorted_neurons_{L2}_last_token.pkl"
            save_path_ap_scores = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/ap_scores_{L2}_last_token.pkl"
        save_as_pickle(save_path_sorted_neurons, sorted_neurons)
        save_as_pickle(save_path_ap_scores, ap_scores)
        
        print(L2, "\n")
        for neuron in sorted_neurons[:10]:
            print(ap_scores[neuron])
    
    del model
    torch.cuda.empty_cache()