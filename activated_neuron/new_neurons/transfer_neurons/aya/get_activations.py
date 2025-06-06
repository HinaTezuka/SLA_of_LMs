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
langs = ['vi', 'ru', 'fr']
num_sentences = 1000
start_indics = {
    "ja": 0,
    "nl": 1000,
    "ko": 2000,
    "it": 3000,
    "en": 4000,
    "vi": 5000,
    "ru": 6000,
    "fr": 7000,
}
multilingual_sentences = multilingual_dataset_for_lang_specific_detection(langs, num_sentences)
print(f"len_multilingual_sentences: {len(multilingual_sentences)}")

# model and tokenizer.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "CohereForAI/aya-expanse-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model_langs = ["ja", "nl", "ko", "it"]

""" get activaitons and save as npz and pkl. """
for L2 in model_langs:
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
        )
    # saving as pkl file.
    save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/aya/activations/{L2}_last_token"
    save_np_arrays(save_path_activations, activations)

    # clean cache.
    torch.cuda.empty_cache()