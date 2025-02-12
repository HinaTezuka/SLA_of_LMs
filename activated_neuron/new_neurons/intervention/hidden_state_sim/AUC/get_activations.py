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
# LLaMA-3
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
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
    total_sentence_num = 5000
    num_sentences = 2000
    dataset = dataset.select(range(total_sentence_num))
    tatoeba_data = []
    for sentence_idx, item in enumerate(dataset):
        if sentence_idx == num_sentences: break
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
    for sentence_idx, item in enumerate(dataset):
        if sentence_idx == num_sentences: break
        if dataset['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
            random_data.append((dataset["translation"][num_sentences+sentence_idx][L1], item["translation"][L2]))
    
    # """ tatoeba translation corpus """
    # dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # # select first 2000 sentences
    # num_sentences = 2000
    # dataset = dataset.select(range(num_sentences))
    # tatoeba_data = []
    # for item in dataset:
    #     # check if there are empty sentences.
    #     if item['translation'][L1] != '' and item['translation'][L2] != '':
    #         tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
    # # tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]
    # tatoeba_data_len = len(tatoeba_data)

    # """
    # baseとして、対訳関係のない1文ずつのペアを作成
    # (L1(en)はhttps://huggingface.co/datasets/agentlans/high-quality-english-sentences,
    # L2はtatoebaの該当データを使用)
    # """
    # random_data = []
    # # L1(en)
    # en_base_ds = load_dataset("agentlans/high-quality-english-sentences")
    # random_data_en = en_base_ds["train"][:num_sentences]
    # en_base_ds_idx = 0
    # for item in dataset:
    #     random_data.append((random_data_en["text"][en_base_ds_idx], item["translation"][L2]))
    #     en_base_ds_idx += 1

    """ tracking neurons """
    activation_types = ["abs", "product"]
    # activation_type = "abs"
    # activation_type = "product"

    for activation_type in activation_types:
        # 対訳ペア
        activation_dict_same_semantics = track_neurons_with_text_data(model, device, 'llama', tokenizer, tatoeba_data, True, activation_type)
        # 非対訳ペア
        activation_dict_non_same_semantics = track_neurons_with_text_data(model, device, 'llama', tokenizer, random_data, False, activation_type)

        # delete some cache
        del model
        torch.cuda.empty_cache()

        """
        (初回だけ)pickleでfileにshared_neurons(track_dict)を保存
        freq dict: (2000対訳ペアを入れた時の） 各種ニューロンの発火頻度
        sum dict: 各種ニューロンの発火値の合計
        """
        # active_THRESHOLD = 0.01

        # translation pair (same_semantics)
        pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/same_semantics/en_{L2}_revised.pkl"
        save_as_pickle(pkl_file_path, activation_dict_same_semantics)
        print(f"pickle file saved: activation_dict(same_semantics): en_{L2}.")

        # non translation pair (non_same_semantics)
        pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/non_same_semantics/en_{L2}_revised.pkl"
        save_as_pickle(pkl_file_path, activation_dict_non_same_semantics)
        print(f"pickle file saved: activation_dict(non_same_semantics): en_{L2}.")

        # """ pickle file(shared_neurons)の解凍/読み込み """
        # with open(pkl_file_path, "rb") as f:
        #     loaded_dict = pickle.load(f)
        # print("unfold pickle")
        # print(loaded_dict[2019][31])
        # sys.exit()
