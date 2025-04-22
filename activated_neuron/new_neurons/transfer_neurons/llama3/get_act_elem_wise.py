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
    track_neurons_with_text_data_elem_wise,
    save_as_pickle,
    compute_ap_and_sort,
    unfreeze_pickle,
    save_np_arrays,
)

# load qa data and shuffle.
qa_num = 1000
qa = load_dataset('apple/mkqa')['train']
qa = qa.shuffle(seed=42)

# model and tokenizer.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", "CohereForAI/aya-expanse-8b"]
model_langs = ["ja", "nl", "ko", "it"]

""" get activaitons and save as npz and pkl. """
for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    for L2 in model_langs:
        # get activations and corresponding labels.
        activations = track_neurons_with_text_data_elem_wise(
            model, 
            device, 
            tokenizer, 
            qa,
            qa_num,
            L2,
            )
        # save activations as npz file.
        model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
        save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/activations/{L2}_last_token_elem_wise"
        save_np_arrays(save_path_activations, activations)
        print(f'saving completed: {model_type, L2}')
        # clean cache.
        torch.cuda.empty_cache()