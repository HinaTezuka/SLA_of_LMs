import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/gpt2")

import numpy as np
import torch
import dill as pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from neuron_detection_funcs import (
    track_neurons_with_text_data,
    save_as_pickle,
    unfreeze_pickle,
)
from visualization_funcs import (
    visualize_neurons_with_line_plot,
)

""" parameters setting """
# GPT-2-small
model_names = {
    # "base": "gpt2",
    # "ja": "rinna/japanese-gpt2-small", # ja
    # "de": "ml6team/gpt2-small-german-finetune-oscar", # ger
    "nl": "GroNLP/gpt2-small-dutch", # du
    "it": "GroNLP/gpt2-small-italian", # ita
    "fr": "dbddv01/gpt2-french-small", # fre
    "ko": "skt/kogpt2-base-v2", # ko
    "es": "datificate/gpt2-small-spanish" # spa
}
device = "cuda" if torch.cuda.is_available() else "cpu"
L1 = "en" # L1 is fixed to english.
active_THRESHOLD = 0.1

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
    neuron_detection_dict_vis, freq_dict, act_sum_dict = track_neurons_with_text_data(model, 'gpt2', device, tokenizer, tatoeba_data, active_THRESHOLD)
    neuron_detection_base_dict_vis, freq_base_dict, act_sum_base_dict = track_neurons_with_text_data(model, 'gpt2', device, tokenizer, random_data, active_THRESHOLD)

    # delete some cache
    del model
    torch.cuda.empty_cache()

    """ for visualization """
    # 各文ペア、各層、各ニューロンの発火ニューロン数
    activated_neurons_L1_vis = neuron_detection_dict_vis["activated_neurons_L1"]
    activated_neurons_L2_vis = neuron_detection_dict_vis["activated_neurons_L2"]
    shared_neurons_vis = neuron_detection_dict_vis["shared_neurons"]
    specific_neurons_L1_vis = neuron_detection_dict_vis["specific_neurons_L1"]
    specific_neurons_L2_vis = neuron_detection_dict_vis["specific_neurons_L2"]
    # for baseline
    shared_neurons_base_vis = neuron_detection_base_dict_vis["shared_neurons"]

    """
    (初回だけ)pickleでfileにshared_neurons(track_dict)を保存
    freq dict: (2000対訳ペアを入れた時の） 各種ニューロンの発火頻度
    sum dict: 各種ニューロンの発火値の合計
    """
    # # 対訳ペア(freq_dict)
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/pickles/same_semantics/act_freq/{active_THRESHOLD}_th/en_{L2}.pkl"
    save_as_pickle(pkl_file_path, freq_dict)
    print("pickle file saved: freq_dict.")
    # 対訳ペア(act_sum_dict)
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/pickles/same_semantics/act_sum/{active_THRESHOLD}_th/en_{L2}.pkl"
    save_as_pickle(pkl_file_path, act_sum_dict)
    print("pickle file saved: act_sum_dict.")
    # 非対訳ペア(freq_base_dict)
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/pickles/non_same_semantics/act_freq/{active_THRESHOLD}_th/en_{L2}.pkl"
    save_as_pickle(pkl_file_path, freq_base_dict)
    print("pickle file saved: freq_base_dict.")
    # 非対訳ペア(act_sum_base_dict)
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/pickles/non_same_semantics/act_sum/{active_THRESHOLD}_th/en_{L2}.pkl"
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
                                        f"{active_THRESHOLD}_th",
                                        # base line
                                        shared_neurons_base_vis,
                                    )

if __name__ == "__main__":
    print('visualization completed.')

