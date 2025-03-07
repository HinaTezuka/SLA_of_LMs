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
    monolingual_dataset,
    get_hidden_states_en_only,
    get_centroid_of_shared_space,
    save_as_pickle,
    unfreeze_pickle,
)

langs = ["ja", "nl", "it", "ko"]
num_sentences = 2000
# LLaMA3-8B
model_name = "mistralai/Mistral-7B-v0.3"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
num_layers = 32

for L2 in langs:
    # centroids of english texts.
    centroids = {}
    # hidden_states for english
    # sentence_pairs = monolingual_dataset(L2, num_sentences)
    sentences = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_mono_train.pkl")
    
    # get centroids of hidden_states(of en-L2 sentence pairs).
    c_hidden_states = get_hidden_states_en_only(model, tokenizer, device, num_layers, sentences)
    shared_space_centroids = get_centroid_of_shared_space(c_hidden_states) # list: [c_layer1, c_layer2, ...]

    # save centroids as pkl.
    save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mistral/centroids/c_mono_sentences_{L2}.pkl"
    save_as_pickle(save_path, shared_space_centroids)