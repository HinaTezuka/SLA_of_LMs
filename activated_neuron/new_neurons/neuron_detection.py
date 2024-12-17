""" neurons detection """
import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons")
import dill as pickle

import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from neuron_detection_funcs import (
    track_neurons_with_text_data,
    save_as_pickle,
)
from visualization_funcs import visualize_neurons_with_line_plot

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

L1 = "en" # L1 is fixed to english.

for L2, model_name in model_names.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

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
    # 対訳ペア
    neuron_detection_dict, neuron_detection_dict_vis, freq_dict, act_sum_dict = track_neurons_with_text_data(model, 'llama', tokenizer, tatoeba_data)
    # 非対訳ペア
    _, neuron_detection_base_dict_vis, freq_base_dict, act_sum_base_dict = track_neurons_with_text_data(model, 'llama', tokenizer, random_data)

    # delete some cache
    del model
    torch.cuda.empty_cache()

    """ for visualization """
    # for main
    # 各文ペア、各層、各ニューロンの発火ニューロン数
    activated_neurons_L1_vis = neuron_detection_dict_vis["activated_neurons_L1"]
    activated_neurons_L2_vis = neuron_detection_dict_vis["activated_neurons_L2"]
    shared_neurons_vis = neuron_detection_dict_vis["shared_neurons"]
    specific_neurons_L1_vis = neuron_detection_dict_vis["specific_neurons_L1"]
    specific_neurons_L2_vis = neuron_detection_dict_vis["specific_neurons_L2"]
    # for base line
    shared_neurons_base_vis = neuron_detection_base_dict_vis["shared_neurons"]

    """
    (初回だけ)pickleでfileにshared_neurons(track_dict)を保存
    freq dict: (2000対訳ペアを入れた時の） 各種ニューロンの発火頻度
    sum dict: 各種ニューロンの発火値の合計
    """
    active_THRESHOLD = 0.01
    # 対訳ペア(freq_dict)
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/same_semantics/act_freq/{active_THRESHOLD}_th/en_{L2}.pkl"
    save_as_pickle(pkl_file_path, freq_dict)
    print("pickle file saved: freq_dict.")
    # 対訳ペア(act_sum_dict)
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/same_semantics/act_sum/{active_THRESHOLD}_th/en_{L2}.pkl"
    save_as_pickle(pkl_file_path, act_sum_dict)
    print("pickle file saved: act_sum_dict.")
    # 非対訳ペア(freq_base_dict)
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/non_same_semantics/act_freq/{active_THRESHOLD}_th/en_{L2}.pkl"
    save_as_pickle(pkl_file_path, freq_base_dict)
    print("pickle file saved: freq_base_dict.")
    # 非対訳ペア(act_sum_base_dict)
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/non_same_semantics/act_sum/{active_THRESHOLD}_th/en_{L2}.pkl"
    save_as_pickle(pkl_file_path, act_sum_base_dict)
    print("pickle file saved: act_sum_base_dict.")

    """ pickle file(shared_neurons)の解凍/読み込み """
    # with open(pkl_file_path, "rb") as f:
    #     loaded_dict = pickle.load(f)
    # print("unfold pickle")
    # print(loaded_dict)
    # sys.exit()

    """ visualization """
    visualize_neurons_with_line_plot(
                                        L1,
                                        L2,
                                        # main
                                        activated_neurons_L1_vis,
                                        activated_neurons_L2_vis,
                                        shared_neurons_vis,
                                        specific_neurons_L1_vis,
                                        specific_neurons_L2_vis,
                                        "0.01_th",
                                        # base line
                                        shared_neurons_base_vis,
                                    )

if __name__ == "__main__":
    print('visualization completed.')
