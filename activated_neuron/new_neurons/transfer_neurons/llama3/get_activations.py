import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
    save_np_arrays,
)

# making multilingual data.
langs = ["ja", "nl", "ko", "it", "en"]
num_sentences = 1000
# start_indics = {
#     "ja": 0,
#     "nl": 500,
#     "ko": 1000,
#     "it": 1500,
#     "en": 2000,
# }
start_indics = {
    "ja": 0,
    "nl": 1000,
    "ko": 2000,
    "it": 3000,
    "en": 4000,
}
multilingual_sentences = multilingual_dataset_for_lang_specific_detection(langs, num_sentences) # 2500 sentences(500 for each lang).
print(f"len_multilingual_sentences: {len(multilingual_sentences)}")

# model and tokenizer.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
is_last_token_onlys = [True, False]
is_last_token_onlys = [True]
model_langs = ["ja", "nl", "ko", "it"]

""" get activaitons and save as npz and pkl. """
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

        # save activations as pickle file.
        if not is_last_token_only:
            save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/activations/{L2}"
            save_path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/labels/{L2}.pkl"
        if is_last_token_only:
            save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/activations/{L2}_last_token_up_proj"
            save_path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/labels/{L2}_last_token_up_proj.pkl"
        print("Started Saving....")
        save_np_arrays(save_path_activations, activations)
        save_as_pickle(save_path_labels, labels)
        print(f"successfully saved activations and labels of {L2} model as pkl, is_last_token_only:{is_last_token_only}.")
        # clean cache.
        del activations, labels
        torch.cuda.empty_cache()