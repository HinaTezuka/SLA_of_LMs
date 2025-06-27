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

# model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", "CohereForAI/aya-expanse-8b"]
model_names = ["CohereForAI/aya-expanse-8b", "mistralai/Mistral-7B-v0.3", "meta-llama/Meta-Llama-3-8B"]
device = "cuda" if torch.cuda.is_available() else "cpu"
# langs = ["ja", "nl", "ko", "it", "en"]
langs = ["ja", "nl", "ko", "it"]
# langs = ['vi', 'ru' 'fr']
score_types = ["cos_sim", "L2_dis"]
# is_reverses = [True, False]
is_reverses = [False] # fix to type-1.

num_candidate_layers = 20
""" candidate neurons. """
candidates = {}
for layer_idx in range(num_candidate_layers):
    for neuron_idx in range(14336):
        candidates.setdefault(layer_idx, []).append(neuron_idx)


for score_type in score_types:
    for model_name in model_names:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
        # calc scores.
        for L2 in langs:
            # monolingual_sentences = monolingual_dataset(L2, num_sentences)
            monolingual_sentences = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_mono_train.pkl")
            # centroids(en-only).
            c_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_en.pkl"
            centroids = unfreeze_pickle(c_path)
            # scores: {(layer_idx, neuron_idx): score, ....}
            # scores = compute_scores_optimized(model, tokenizer, device, monolingual_sentences, candidates, centroids[L2], score_type)
            scores = compute_scores_optimized(model, tokenizer, device, monolingual_sentences, candidates, centroids, score_type) # for en only.
            # 降順
            # sorted_neurons = [neuron for neuron, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)] # original list
            sorted_neurons, score_dict = sort_neurons_by_score(scores) # np用
            
            # save as pkl.
            # sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
            # score_dict_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_score_dict_mono_train.pkl"
            sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/en_only_mono_train_{L2}.pkl"
            score_dict_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/en_only_score_dict_mono_train{L2}.pkl"
            save_as_pickle(sorted_neurons_path, sorted_neurons)
            save_as_pickle(score_dict_path, score_dict)
            print(f"saved scores for: {L2}.")
            
            # clean cache.
            del scores, sorted_neurons, score_dict
            torch.cuda.empty_cache()
    
        # celan cache.
        del model
        torch.cuda.empty_cache()