import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/gpt2")

import numpy as np
import torch
import dill as pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from neuron_detection_funcs import track_neurons_with_text_data
from visualization_funcs import visualize_neurons_with_line_plot_mean

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

L1 = "en" # L1 is fixed to english.

for L2, model_name in model_names.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    """ tatoeba translation corpus """
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # select first ? sentences
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
    neuron_detection_dict, neuron_detection_dict_vis, freq_dict, act_sum_dict = track_neurons_with_text_data(model, 'gpt2', tokenizer, tatoeba_data, 0, 0)
    _, neuron_detection_base_dict_vis, freq_base_dict, _ = track_neurons_with_text_data(model, 'gpt2', tokenizer, random_data, 0, 0)

    # delete some cache
    del model
    torch.cuda.empty_cache()

    # for visualization
    # 各文ペア、各層、各ニューロンの発火ニューロン数
    activated_neurons_L1_vis = neuron_detection_dict_vis["activated_neurons_L1"]
    activated_neurons_L2_vis = neuron_detection_dict_vis["activated_neurons_L2"]
    shared_neurons_vis = neuron_detection_dict_vis["shared_neurons"]
    specific_neurons_L1_vis = neuron_detection_dict_vis["specific_neurons_L1"]
    specific_neurons_L2_vis = neuron_detection_dict_vis["specific_neurons_L2"]

    """ for base line """
    # for visualization
    activated_neurons_L1_base_vis = neuron_detection_base_dict_vis["activated_neurons_L1"]
    activated_neurons_L2_base_vis = neuron_detection_base_dict_vis["activated_neurons_L2"]
    shared_neurons_base_vis = neuron_detection_base_dict_vis["shared_neurons"]
    specific_neurons_L1_base_vis = neuron_detection_base_dict_vis["specific_neurons_L1"]
    specific_neurons_L2_base_vis = neuron_detection_base_dict_vis["specific_neurons_L2"]

    """ save pickle(act_sum_dict) """
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/gpt2/pickles/tatoeba_0_th/act_sum_dict/act_sum_dict_en_{L2}_tatoeba_0_th.pkl"
    # directoryを作成（存在しない場合のみ)
    os.makedirs(os.path.dirname(pkl_file_path), exist_ok=True)
    with open(pkl_file_path, "wb") as f:
        pickle.dump(act_sum_dict, f)
    print("pickle file saved.")

    """ save pickle(act_freq_dict) """
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/gpt2/pickles/tatoeba_0_th/act_freq_dict/act_freq_dict_en_{L2}_tatoeba_0_th.pkl"
    # directoryを作成（存在しない場合のみ)
    os.makedirs(os.path.dirname(pkl_file_path), exist_ok=True)
    with open(pkl_file_path, "wb") as f:
        pickle.dump(freq_dict, f)
    print("pickle file saved.")

    """ save pickle(act_freq_base_dict) """
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/gpt2/pickles/tatoeba_0_th/act_freq_base_dict/act_freq_base_dict_en_{L2}_tatoeba_0_th.pkl"
    # directoryを作成（存在しない場合のみ)
    os.makedirs(os.path.dirname(pkl_file_path), exist_ok=True)
    with open(pkl_file_path, "wb") as f:
        pickle.dump(freq_base_dict, f)
    print("pickle file saved.")


    """ visualization """
    # visualize_neurons_with_line_plot_mean(
    #                                     L1,
    #                                     L2,
    #                                     # main
    #                                     activated_neurons_L1_vis,
    #                                     activated_neurons_L2_vis,
    #                                     shared_neurons_vis,
    #                                     specific_neurons_L1_vis,
    #                                     specific_neurons_L2_vis,
    #                                     "tatoeba_0_th",
    #                                     # base line
    #                                     shared_neurons_base_vis,
    #                                 )

