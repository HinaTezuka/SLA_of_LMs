import sys
import dill as pickle

import dgh
import numpy as np
import matplotlib.pyplot as plt
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.spatial.distance import directed_hausdorff

from funcs import (
    get_hidden_states,
    save_as_pickle,
)

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

for L2, model_name in model_names.items():
    L1 = "en" # L1 is fixed to english.

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    
    """ tatoeba translation corpus """
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # select first 2000 sentences
    num_sentences = 2
    dataset = dataset.select(range(num_sentences))
    tatoeba_data = []
    for item in dataset:
        # check if there are empty sentences.
        if item['translation'][L1] != '' and item['translation'][L2] != '':
            tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
    tatoeba_data_len = len(tatoeba_data)

    """
    making non-translation pair for baseline.
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
    ht_same = get_hidden_states(model, tokenizer, device, tatoeba_data) # hidden_states for same semantic pairs
    ht_diff = get_hidden_states(model, tokenizer, device, random_data) # hidden_states for different semantic pairs

    """ compute Gromov-Hausdorff Distances. """
    def compute_dgh(hidden_states_dict: dict):
        """
        compute Gromov-Hausdorff Distance using: https://github.com/vlad-oles/dgh.
        """
        results = {}
        for sentence_idx, hidden_states in hidden_states_dict.items():
            results_layer = [] # gh_distance per layer.
            # hidden_states: (L1_hidden_states, L2_hidden_states)
            for layer_idx in range(32):
                # L1_hidden_states: hidden_states[0], L2_hidden_states: hidden_states[1].
                X = hidden_states[0][layer_idx]  # shape: (6, 4096)
                Y = hidden_states[1][layer_idx]  # shape: (6, 4096)

                # X → Y の Hausdorff dis
                H_XY = directed_hausdorff(X, Y)[0]
                # Y → X の Hausdorff dis
                H_YX = directed_hausdorff(Y, X)[0]
                d = (H_XY + H_YX) / 2
                # gh_dis = dgh.upper(H_XY, H_YX)
                results_layer.append(d)
                print((H_XY, H_YX))
            sys.exit()
            results[sentence_idx] = results_layer
            print(results[sentence_idx])
            sys.exit()
    
    print(compute_dgh(ht_same))