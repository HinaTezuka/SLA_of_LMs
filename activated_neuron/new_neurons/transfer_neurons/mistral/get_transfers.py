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
    save_as_pickle,
    unfreeze_pickle,
)

# Mistral
model_name = "mistralai/Mistral-7B-v0.3"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_sentences = 500
langs = ["ja", "nl", "ko", "it"]
score_types = ["L2_dis", "cos_sim"]

""" candidate neurons. """
candidates = {}
for layer_idx in range(16):
    for neuron_idx in range(14336):
        candidates.setdefault(layer_idx, []).append(neuron_idx)

# get centroids.
c_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mistral/centroids/c.pkl"
centroids = unfreeze_pickle(c_path)

# calc scores.
for L2 in langs:
    # L2-specific neurons
    # save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/sorted_neurons_{L2}.pkl"
    # save_path_ap_scores = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/ap_scores_{L2}.pkl"
    # sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
    # ap_scores = unfreeze_pickle(save_path_ap_scores)

    monolingual_sentences = monolingual_dataset(L2, num_sentences)
    for score_type in score_types:
        # scores: {(layer_idx, neuron_idx): score, ....}
        scores = compute_scores(model, tokenizer, device, monolingual_sentences, candidates, centroids[L2], score_type)
        # 降順
        sorted_neurons = [neuron for neuron, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
        
        # save as pkl.
        path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mistral/final_scores/{score_type}/{L2}.pkl"
        save_as_pickle(path, sorted_neurons)
        print("saved scores for: {L2}.")