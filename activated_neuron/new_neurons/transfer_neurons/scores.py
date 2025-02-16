"""
calc scores for each lang-specific neuron.
"""
import sys
import pickle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    monolingual_dataset,
    compute_scores,
    save_as_pickle,
    unfreeze_pickle,
)

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
num_sentences = 500

# get centroids.
c_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/centroids/c1.pkl"
centroids = unfreeze_pickle(c_path)

# calc scores.
for L2, model_name in model_names.items():
    # L2-specific neurons
    save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/sorted_neurons_{L2}.pkl"
    save_path_ap_scores = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/ap_scores_{L2}.pkl"
    sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
    ap_scores = unfreeze_pickle(save_path_ap_scores)

    monolingual_sentences = monolingual_dataset(L2, num_sentences)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    scores = compute_scores(model, tokenizer, device, monolingual_sentences, neurons, centroids[L2])
