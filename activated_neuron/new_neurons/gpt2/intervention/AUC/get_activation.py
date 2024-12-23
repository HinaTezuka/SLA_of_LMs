""" neurons detection """
import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/hidden_state_sim/AUC")
import dill as pickle

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
# GPT-2-small
model_names = {
    # "base": "gpt2",
    "ja": "rinna/japanese-gpt2-small", # ja
    # "de": "ml6team/gpt2-small-german-finetune-oscar", # ger
    "nl": "GroNLP/gpt2-small-dutch", # du
    "it": "GroNLP/gpt2-small-italian", # ita
    "fr": "dbddv01/gpt2-french-small", # fre
    "ko": "skt/kogpt2-base-v2", # ko
    "es": "datificate/gpt2-small-spanish" # spa
}
device = "cuda" if torch.cuda.is_available() else "cpu"

L1 = "en" # L1 is fixed to english.

for L2, model_name in model_names.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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
    baseとして、対訳関係のない1文ずつのペアを作成
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

    """ tracking neurons """
    activation_type = "abs"
    # activation_type = "product"

    # 対訳ペア
    activation_dict_same_semantics = track_neurons_with_text_data(model, device, 'gpt2', tokenizer, tatoeba_data, True, activation_type)
    # 非対訳ペア
    activation_dict_non_same_semantics = track_neurons_with_text_data(model, device, 'gpt2', tokenizer, random_data, False, activation_type)

    # delete some cache
    del model
    torch.cuda.empty_cache()

    """
    (初回だけ)pickleでfileにshared_neurons(track_dict)を保存
    freq dict: (2000対訳ペアを入れた時の） 各種ニューロンの発火頻度
    sum dict: 各種ニューロンの発火値の合計
    """

    # translation pair (same_semantics)
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/pickles/AUC/act_{activation_type}/same_semantics/en_{L2}.pkl"
    save_as_pickle(pkl_file_path, activation_dict_same_semantics)
    print(f"pickle file saved: activation_dict(same_semantics): en_{L2}.")

    # non translation pair (non_same_semantics)
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/pickles/AUC/act_{activation_type}/non_same_semantics/en_{L2}.pkl"
    save_as_pickle(pkl_file_path, activation_dict_non_same_semantics)
    print(f"pickle file saved: activation_dict(non_same_semantics): en_{L2}.")

    # """ pickle file(shared_neurons)の解凍/読み込み """
    # with open(pkl_file_path, "rb") as f:
    #     loaded_dict = pickle.load(f)
    # print("unfold pickle")
    # print(loaded_dict[2019][31])
    # sys.exit()
