"""
calc centroid.
centroid: center point of Language-Agnostic Space.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    monolingual_dataset_en,
    get_hidden_states_en_only,
    get_centroid_of_shared_space,
    save_as_pickle,
    unfreeze_pickle,
)

langs = ["ja", "nl", "ko", "it", "en", 'vi', 'ru', 'fr']
model_names =  ['bigscience/bloom-3b']
device = "cuda" if torch.cuda.is_available() else "cpu"

for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya' if 'aya' in model_name else 'bloom'
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    num_layers = 32 if model_type in ['llama3', 'mistral', 'aya'] else 30
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # centroids of english texts.
    centroids = {} # { L2: [shared_centroids(en-L2)_1, ...} <- len(values) = layer_num.

    for L2 in langs:
        # hidden_states for each L2.
        # mono L2 sentences.
        path_mono_train = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_mono_train.pkl"
        sentences = unfreeze_pickle(path_mono_train)
        # get centroids of hidden_states(of en-L2 sentence pairs).
        c_hidden_states = get_hidden_states_en_only(model, tokenizer, device, num_layers, sentences)
        shared_space_centroids = get_centroid_of_shared_space(c_hidden_states) # list: [c_layer1, c_layer2, ...]

        # save centroids as pkl.
        save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_{L2}.pkl"
        save_as_pickle(save_path, shared_space_centroids)
    
    del model
    torch.cuda.empty_cache()