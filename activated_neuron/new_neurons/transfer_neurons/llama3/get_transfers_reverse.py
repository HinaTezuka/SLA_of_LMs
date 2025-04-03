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

# LLaMA3-8B
model_name = "meta-llama/Meta-Llama-3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
langs = ["ja", "nl", "ko", "it"]
langs = ['en']
score_types = ["cos_sim", "L2_dis"]

""" candidate neurons. """
candidates = {}
for layer_idx in range(32):
    for neuron_idx in range(14336):
        candidates.setdefault(layer_idx, []).append(neuron_idx)

# calc scores.
for L2 in langs:
    c_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/centroids/c_train_{L2}.pkl"
    centroids = unfreeze_pickle(c_path)
    monolingual_sentences = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_mono_train.pkl")
    for score_type in score_types:
        # scores: {(layer_idx, neuron_idx): score, ....}
        scores = compute_scores_optimized(model, tokenizer, device, monolingual_sentences, candidates, centroids, score_type) # for L2 only: reverse transfers.
        # 降順
        # sorted_neurons = [neuron for neuron, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)] # original list
        sorted_neurons, score_dict = sort_neurons_by_score(scores) # np
        
        # save as pkl.
        sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
        score_dict_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/reverse/{score_type}/{L2}_score_dict.pkl"
        save_as_pickle(sorted_neurons_path, sorted_neurons)
        save_as_pickle(score_dict_path, score_dict)
        print(f"saved scores for: {L2}.")
        
        del scores, sorted_neurons, score_dict
        torch.cuda.empty_cache()