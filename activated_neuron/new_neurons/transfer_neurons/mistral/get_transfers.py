"""
calc scores for each lang-specific neuron.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    monolingual_dataset,
    compute_scores,
    compute_scores_optimized,
    sort_neurons_by_score,
    save_as_pickle,
    unfreeze_pickle,
)

# Mistral
model_name = "mistralai/Mistral-7B-v0.3"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_sentences = 1000
langs = ["ja", "nl", "ko", "it"]
# langs = ["it", "nl", "ja", "ko"]
score_types = ["cos_sim", "L2_dis"]

""" candidate neurons. """
candidates = {}
for layer_idx in range(21):
    for neuron_idx in range(14336):
        candidates.setdefault(layer_idx, []).append(neuron_idx)

# get centroids.
num_sentences_for_calc_centroids = 2000
c_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mistral/centroids/c_n{num_sentences_for_calc_centroids}.pkl"
centroids = unfreeze_pickle(c_path)

# calc score
for L2 in langs:
    # L2-specific neurons
    # save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/sorted_neurons_{L2}.pkl"
    # save_path_ap_scores = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/ap_scores_{L2}.pkl"
    # sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
    # ap_scores = unfreeze_pickle(save_path_ap_scores)

    monolingual_sentences = monolingual_dataset(L2, num_sentences)
    for score_type in score_types:
        # scores: {(layer_idx, neuron_idx): score, ....}
        scores = compute_scores_optimized(model, tokenizer, device, monolingual_sentences, candidates, centroids[L2], score_type)
        # 降順
        # sorted_neurons = [neuron for neuron, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)] # original list
        sorted_neurons, score_dict = sort_neurons_by_score(scores) # np用
        
        # save as pkl.
        sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mistral/final_scores/{score_type}/{L2}.pkl"
        score_dict_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mistral/final_scores/{score_type}/{L2}_score_dict.pkl"
        save_as_pickle(sorted_neurons_path, sorted_neurons)
        save_as_pickle(score_dict_path, score_dict)
        print("saved scores for: {L2}.")

        del scores, sorted_neurons, score_dict
        torch.cuda.empty_cache()