"""
detect language specific neurons.
"""
import sys
import dill as pickle

import torch

from funcs import (
    unfreeze_pickle,
    compute_ap_and_sort,
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
# indices where each L2 sentences begin.
start_indics = {
    "ja": 0,
    "nl": 500,
    "ko": 1000,
    "it": 1500,
}
num_sentences_per_L2 = 500

for L2, model_name in model_names.items():
    """
    activations for L2.
    activation_dict
    {
        text_idx:
            layer_idx: [(neuron_idx, act_value), (neuron_idx, act_value), ....]
    }
    """
    # unfreeze activations.
    file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/activations/{L2}.pkl"
    activations = unfreeze_pickle(file_path)
    
    # calc AP scores.
    sorted_neurons, ap_scores = compute_ap_and_sort(activations, start_indics[L2], num_sentences_per_L2)

    # save AP scores as pkl.
    save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/sorted_neurons_{L2}.pkl"
    save_path_ap_scores = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/ap_scores_{L2}.pkl"
    save_as_pickle(save_path_sorted_neurons, sorted_neurons)
    save_as_pickle(save_path_ap_scores, ap_scores)
    
    print(L2, "\n")
    for neuron in sorted_neurons[:10]:
        print(ap_scores[neuron])