"""
detect language specific neurons
"""
import sys
# import dill as pickle
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
num_sentences = 1000
multilingual_sentences = multilingual_dataset_for_lang_specific_detection(langs, num_sentences) # 2500 sentences(500 for each lang).
print(f"len_multilingual_sentences: {len(multilingual_sentences)}")

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
start_indics = {
    "ja": 0,
    "nl": 1000,
    "ko": 2000,
    "it": 3000,
    "en": 4000,
}
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

    # save activations as pickle file.
    if not is_last_token_only:
        save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/activations/{L2}_normal.pkl"
        save_path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/labels/{L2}_normal.pkl"
    if is_last_token_only:
        save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/activations/{L2}_last_token.pkl"
        save_path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/labels/{L2}_last_token.pkl"
    save_as_pickle(save_path_activations, activations)
    save_as_pickle(save_path_labels, labels)
    print(f"successfully saved activations and labels of {L2} model as pkl.")

    # clean cache.
    del activations, labels
    del model
    torch.cuda.empty_cache()