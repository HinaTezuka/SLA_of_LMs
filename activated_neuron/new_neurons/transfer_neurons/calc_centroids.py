"""
calc centroid.
centroid: center point of Language-Agnostic Space.
"""
import sys
# import dill as pickle
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
)

# making multilingual data.
langs = ["ja", "nl", "ko", "it"]
num_sentences_per_L2 = 1000
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
num_layers = 32
# centroids of L2s
centroids = {} # { L2: shared_centroids(en-L2), ...}

# hidden_states and get centroids per L2.
for L2, model_name in model_names.items():
    sentence_pairs = multilingual_dataset_for_centroid_detection(langs, num_sentences_per_L2)
    # model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # text data for L2.
    data = sentence_pairs[L2]
    # get centroids of hidden_states(of en-L2 sentence pairs).
    c_hidden_states = get_hidden_states(model, tokenizer, device, num_layers, data)
    shared_space_centroids = get_centroid_of_shared_space(c_hidden_states)
    centroids[L2] = shared_space_centroids

    # clear caches.
    del model
    torch.cuda.empty_cache()

# save centroids as pkl.
save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/centroids/c1.pkl"
save_as_pickle(save_path, centroids)