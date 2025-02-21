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
    get_hidden_states,
    get_centroid_of_shared_space,
    # get_centroids_per_L2,
    # get_centroids_of_shared_space,
    save_as_pickle,
)

num_sentences = 2000
# LLaMA3-8B.
model_name = "meta-llama/Meta-Llama-3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
num_layers = 32

# centroids of english texts.
centroids = {} # { L2: [shared_centroids(en-L2)_1, ...} <- len(values) = 32(layer_num).

# hidden_states for english
sentence_pairs = multilingual_dataset_for_centroid_detection(langs, num_sentences_per_L2)
# text data for L2.
data = sentence_pairs[L2]
# get centroids of hidden_states(of en-L2 sentence pairs).
c_hidden_states = get_hidden_states(model, tokenizer, device, num_layers, data)
shared_space_centroids = get_centroid_of_shared_space(c_hidden_states)
centroids[L2] = shared_space_centroids

# save centroids as pkl.
save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/centroids/c_n{num_sentences_per_L2}_en.pkl"
save_as_pickle(save_path, centroids)