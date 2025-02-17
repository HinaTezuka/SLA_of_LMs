"""
detect language specific neurons.
"""
import sys
import pickle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    multilingual_dataset_for_lang_specific_detection,
    track_neurons_with_text_data,
    save_as_pickle,
    compute_ap_and_sort,
    unfreeze_pickle,
)

# making multilingual data.
langs = ["ja", "nl", "ko", "it", "en"]
num_sentences = 500
start_indics = {
    "ja": 0,
    "nl": 500,
    "ko": 1000,
    "it": 1500,
    "en": 2000,
}
# start_indics = {
#     "ja": 0,
#     "nl": 100,
#     "ko": 200,
#     "it": 300,
#     "en": 400,
# }
multilingual_sentences = multilingual_dataset_for_lang_specific_detection(langs, num_sentences) # 2500 sentences(500 for each lang).
# LLaMA-3(8B) models.
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}
device = "cuda" if torch.cuda.is_available() else "cpu"
is_last_token_only = True

""" get activaitons and save as pkl. """
for L2, model_name in model_names.items():
    # model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # start and end indices.
    start_idx = start_indics[L2]
    end_idx = start_idx + num_sentences
    # get activations and corresponding labels.
    activations, labels = track_neurons_with_text_data(
        model, 
        device, 
        tokenizer, 
        multilingual_sentences,
        start_idx,
        end_idx, 
        is_last_token_only,
        )
    # calc AP scores.
    sorted_neurons, ap_scores = compute_ap_and_sort(activations, labels)
    # clean caches.
    del activations, labels

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

    # clean cache.
    del model
    torch.cuda.empty_cache()