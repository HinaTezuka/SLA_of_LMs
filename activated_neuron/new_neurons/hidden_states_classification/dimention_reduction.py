import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/hidden_states_classification")
import dill as pickle
from collections import defaultdict
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from funcs import (
    get_hidden_states,
    get_hidden_states_intervention,
    plot_pca,
    plot_plsr,
    plot_umap,
    save_as_pickle,
    unfreeze_pickle,
)

""" extract hidden states (only for last token) and make inputs for the model. """

""" model configs """
# LLaMA-3(8B)
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}
device = "cuda" if torch.cuda.is_available() else "cpu"
activation_type = "abs"
# activation_type = "product"
norm_type = "no"
top_n = 30000
top_n_for_baseline = 50000

for L2, model_name in model_names.items():
    L1 = "en" # L1 is fixed to english.

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    
    """ tatoeba translation corpus """
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # select first 2000 sentences
    num_sentences = 2000
    dataset = dataset.select(range(num_sentences))
    tatoeba_data = []
    for item in dataset:
        # check if there are empty sentences.
        if item['translation'][L1] != '' and item['translation'][L2] != '':
            tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
    # tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]
    tatoeba_data_len = len(tatoeba_data)

    """
    baselineとして、対訳関係のない1文ずつのペアを作成
    (L1(en)はhttps://huggingface.co/datasets/agentlans/high-quality-english-sentences,
    L2はtatoebaの該当データを使用)
    """
    random_data = []
    # L1(en)
    en_base_ds = load_dataset("agentlans/high-quality-english-sentences")
    random_data_en = en_base_ds["train"][:num_sentences]
    en_base_ds_idx = 0
    for item in dataset:
        random_data.append((random_data_en["text"][en_base_ds_idx], item["translation"][L2]))
        en_base_ds_idx += 1
    
    """ extract hidden states """
    # shape: (num_layers, num_pairs, 8192) <- layerごとに回帰モデルをつくるため.
    # features_label1 = get_hidden_states(model, tokenizer, device, tatoeba_data)
    # features_label0 = get_hidden_states(model, tokenizer, device, random_data)
    
    """ get top AP neurons (layer_idx, neuron_idx) """
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/sorted_neurons_{L2}.pkl"
    AP_list = unfreeze_pickle(pkl_file_path)
    AP_list = AP_list[:top_n]
    # baseline
    AP_baseline = random.sample(AP_list[top_n_for_baseline+1:], len(AP_list[top_n_for_baseline+1:]))
    AP_baseline = AP_baseline[:top_n]

    # intervention of high AP neurons
    features_label1_intervention = get_hidden_states_intervention(model, tokenizer, device, tatoeba_data, AP_list)
    features_label0_intervention = get_hidden_states_intervention(model, tokenizer, device, random_data, AP_list)
    # intervention of baseline
    features_label1_base = get_hidden_states_intervention(model, tokenizer, device, tatoeba_data, AP_baseline)
    features_label0_base = get_hidden_states_intervention(model, tokenizer, device, random_data, AP_baseline)

    # delete model (for saving memory).
    del model

    """ plot with dimention reduction. """
    # normal
    # plot_plsr(features_label1, features_label0, L2)
    # intervention
    plot_plsr(features_label1_intervention, features_label0_intervention, L2, "yes")
    # for baseline
    plot_plsr(features_label1_base, features_label0_base, L2, "base")

    # delete cache
    torch.cuda.empty_cache()
