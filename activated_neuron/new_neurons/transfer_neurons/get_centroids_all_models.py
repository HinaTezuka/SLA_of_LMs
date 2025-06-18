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
    multilingual_dataset_for_centroid_detection,
    get_hidden_states,
    get_centroid_of_shared_space,
    # get_centroids_per_L2,
    # get_centroids_of_shared_space,
    save_as_pickle,
    unfreeze_pickle,
)

# making multilingual data.
langs = ['ja', 'nl', 'ko', 'it', 'vi', 'ru', 'fr']
num_sentences_per_L2 = 2000
# LLaMA3-8B / Mistral-7B / Aya-expanse-8B / BLOOM-3B.
model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b', 'bigscience/bloom-3b']
model_names = ['bigscience/bloom-3b']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hidden_states and get centroids per L2.
for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya' if 'aya' in model_name else 'bloom'
    num_layers = 32 if model_type in ['llama3', 'mistral', 'aya'] else 31
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # centroids of L2
    centroids = {} # { L2: [shared_centroids(en-L2)_1, ...} <- len(values) = layer_num.
    for L2 in langs:
        # sentence_pairs = multilingual_dataset_for_centroid_detection(langs, num_sentences_per_L2)
        sentence_pairs = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_multi_train.pkl")
        # text data for L2.
        # data = sentence_pairs[L2]
        # get centroids of hidden_states(of en-L2 sentence pairs).
        c_hidden_states = get_hidden_states(model, tokenizer, device, num_layers, sentence_pairs)
        shared_space_centroids = get_centroid_of_shared_space(c_hidden_states)
        centroids[L2] = shared_space_centroids

    # save centroids as pkl.
    # save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_vi_ru_fr.pkl"
    # for phi4 and qwen3 only.
    save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_all_langs_en_L2.pkl"
    save_as_pickle(save_path, centroids)

    # clean cache.
    del model
    torch.cuda.empty_cache()