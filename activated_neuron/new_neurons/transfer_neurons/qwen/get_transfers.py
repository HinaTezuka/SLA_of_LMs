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
    compute_scores_optimized_qwen,
    sort_neurons_by_score,
    save_as_pickle,
    unfreeze_pickle,
)

# Qwen3-8B.
model_name = "Qwen/Qwen3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_sentences = 1000
langs = ['ja', 'nl', 'ko', 'it', 'vi', 'ru', 'fr']
score_types = ["cos_sim", "L2_dis"]
num_candidate_layers = 36

""" candidate neurons. """
candidates = {}
for layer_idx in range(num_candidate_layers):
    for neuron_idx in range(12288):
        candidates.setdefault(layer_idx, []).append(neuron_idx)

# get centroids.
# each L2.
# c_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qwen/centroids/c_train.pkl"
c_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qwen/centroids/c_train_all_langs_en_L2.pkl"
# en-only
# c_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qwen/centroids/c_train_en.pkl"
centroids = unfreeze_pickle(c_path)

# calc scores.
for L2 in langs:
    # monolingual_sentences = monolingual_dataset(L2, num_sentences)
    monolingual_sentences = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_mono_train.pkl")
    for score_type in score_types:
        # scores: {(layer_idx, neuron_idx): score, ....}
        scores = compute_scores_optimized_qwen(model, tokenizer, device, monolingual_sentences, candidates, centroids[L2], score_type)
        # scores = compute_scores_optimized(model, tokenizer, device, monolingual_sentences, candidates, centroids, score_type) # for en only.
        # 降順
        # sorted_neurons = [neuron for neuron, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)] # original list
        sorted_neurons, score_dict = sort_neurons_by_score(scores) # np用
        
        # save as pkl.
        sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qwen/final_scores/{score_type}/{L2}_mono_train.pkl"
        score_dict_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qwen/final_scores/{score_type}/{L2}_score_dict_mono_train.pkl"
        # sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qwen/final_scores/{score_type}/en_only_mono_train_{L2}.pkl"
        # score_dict_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qwen/final_scores/{score_type}/en_only_score_dict_mono_train{L2}.pkl"
        save_as_pickle(sorted_neurons_path, sorted_neurons)
        save_as_pickle(score_dict_path, score_dict)
        print(f"saved scores for: {L2}.")
        
        del scores, sorted_neurons, score_dict
        torch.cuda.empty_cache()