""" neurons detection """
import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/hidden_state_sim/AUC")
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from expertise_funcs import (
    track_neurons_with_text_data,
    save_as_pickle,
)

""" models """
# LLaMA-3
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    # "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}
device = "cuda" if torch.cuda.is_available() else "cpu"

L1 = "en" # L1 is fixed to english.

for L2, model_name in model_names.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    """ tatoeba translation corpus """
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # select first 2000 sentences
    total_sentence_num = 2000 if L2 == "ko" else 5000
    num_sentences = 2000
    dataset = dataset.select(range(total_sentence_num))
    tatoeba_data = []
    for sentence_idx, item in enumerate(dataset):
        if sentence_idx == num_sentences: break
        # check if there are empty sentences.
        if item['translation'][L1] != '' and item['translation'][L2] != '':
            tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
    tatoeba_data_len = len(tatoeba_data)

    """
    non-translation pair for baseline.
    """
    random_data = []
    if L2 == "ko": # koreanはデータ数が足りない
        dataset2 = load_dataset("tatoeba", lang1=L1, lang2="ja", split="train").select(range(5000))
    for sentence_idx, item in enumerate(dataset):
        if sentence_idx == num_sentences: break
        if L2 == "ko" and dataset2['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
            random_data.append((dataset2["translation"][num_sentences+sentence_idx][L1], item["translation"][L2])) 
        elif L2 != "ko" and dataset['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
            random_data.append((dataset["translation"][num_sentences+sentence_idx][L1], item["translation"][L2]))

    """ tracking neurons """
    activation_types = ["product", "abs"]

    for activation_type in activation_types:
        # 対訳ペア
        activation_dict_same_semantics = track_neurons_with_text_data(model, device, 'llama', tokenizer, tatoeba_data, True, activation_type)
        # 非対訳ペア
        activation_dict_non_same_semantics = track_neurons_with_text_data(model, device, 'llama', tokenizer, random_data, False, activation_type)

        # delete some cache
        del model
        torch.cuda.empty_cache()

        # translation pair (same_semantics)
        pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/same_semantics/en_{L2}_revised.pkl"
        save_as_pickle(pkl_file_path, activation_dict_same_semantics)
        print(f"pickle file saved: activation_dict(same_semantics): en_{L2} saccessfully saved.")

        # non translation pair (non_same_semantics)
        pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/non_same_semantics/en_{L2}_revised.pkl"
        save_as_pickle(pkl_file_path, activation_dict_non_same_semantics)
        print(f"pickle file saved: activation_dict(non_same_semantics): en_{L2} saccessfully saved.")

        # """ pickle file(shared_neurons)の解凍/読み込み """
        # with open(pkl_file_path, "rb") as f:
        #     loaded_dict = pickle.load(f)
        # print("unfold pickle")
        # print(loaded_dict[2019][31])
        # sys.exit()
