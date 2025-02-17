import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons")
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
num_sentences = 100
# start_indics = {
#     "ja": 0,
#     "nl": 500,
#     "ko": 1000,
#     "it": 1500,
#     "en": 2000,
# }
start_indics = {
    "ja": 0,
    "nl": 100,
    "ko": 200,
    "it": 300,
    "en": 400,
}
multilingual_sentences = multilingual_dataset_for_lang_specific_detection(langs, num_sentences) # 2500 sentences(500 for each lang).
print(f"len_multilingual_sentences: {len(multilingual_sentences)}")

# model and tokenizer.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
is_last_token_onlys = [True, False]
model_langs = ["ja", "nl", "ko", "it"]

""" get activaitons and save as pkl. """
for L2 in model_langs:
    # start and end indices.
    start_idx = start_indics[L2]
    end_idx = start_idx + num_sentences
    for is_last_token_only in is_last_token_onlys:
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

        # save AP scores as pkl.
        if not is_last_token_only:
            save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/ap_lang_specific/sorted_neurons_{L2}.pkl"
            save_path_ap_scores = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/ap_lang_specific/ap_scores_{L2}.pkl"
        elif is_last_token_only:
            save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/ap_lang_specific/sorted_neurons_{L2}_last_token.pkl"
            save_path_ap_scores = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/ap_lang_specific/ap_scores_{L2}_last_token.pkl"
        save_as_pickle(save_path_sorted_neurons, sorted_neurons)
        save_as_pickle(save_path_ap_scores, ap_scores)
        
        #
        print(L2)
        for neuron in sorted_neurons[:10]:
            print(ap_scores[neuron])

        # clean cache.
        del activations, labels, sorted_neurons, ap_scores
        torch.cuda.empty_cache()