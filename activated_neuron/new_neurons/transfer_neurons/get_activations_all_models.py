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
langs = ["ja", "nl", "ko", "it", "en", 'vi', 'ru', 'fr']
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
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", 'CohereForAI/aya-expanse-8b', 'microsoft/phi-4']
# model_names = ['microsoft/phi-4']
model_langs = ['ja', 'nl', 'ko', 'it', 'vi', 'ru', 'fr']

""" get activaitons and save as npz and pkl. """
for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya' if 'aya' in model_name else 'phi4'
    if model_type != 'phi4':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

    for L2 in model_langs:
        # start and end indices.
        start_idx = start_indics[L2]
        end_idx = start_idx + num_sentences
        # get activations and corresponding labels.
        activations, labels = track_neurons_with_text_data(
            model, 
            model_type,
            device, 
            tokenizer, 
            multilingual_sentences, 
            start_idx, 
            end_idx,
            )
        # saving as pkl file.
        save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/activations/{L2}_last_token"
        save_np_arrays(save_path_activations, activations)
    
    # clean cache.
    del model
    torch.cuda.empty_cache()