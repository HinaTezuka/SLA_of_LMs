""" neurons detection """
import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")
# import pickle
import dill as pickle

import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from neuron_detection_funcs import track_neurons_with_text_data
from visualization_funcs import visualize_neurons_with_line_plot_mean, visualize_neurons_with_line_plot, visualize_neurons_with_line_plot_simple, visualize_neurons_with_line_plot_b

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
    # model = AutoModelForCausalLM.from_pretrained(model_name)

    """ tatoeba translation corpus """
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # select first 100 sentences
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
    neuron_detection_dict, neuron_detection_dict_vis, freq_dict, act_sum_dict = track_neurons_with_text_data(model, 'llama', tokenizer, tatoeba_data, 0, 0)
    # 非対訳ペア
    _, neuron_detection_base_dict_vis, freq_base_dict, _ = track_neurons_with_text_data(model, 'llama', tokenizer, random_data, 0, 0)

    # delete some cache
    del model
    torch.cuda.empty_cache()

    """ main """
    #
    activated_neurons_L1 = neuron_detection_dict["activated_neurons_L1"]
    activated_neurons_L2 = neuron_detection_dict["activated_neurons_L2"]
    shared_neurons = neuron_detection_dict["shared_neurons"]
    specific_neurons_L1 = neuron_detection_dict["specific_neurons_L1"]
    specific_neurons_L2 = neuron_detection_dict["specific_neurons_L2"]

    # for visualization
    # 各文ペア、各層、各ニューロンの発火ニューロン数
    activated_neurons_L1_vis = neuron_detection_dict_vis["activated_neurons_L1"]
    activated_neurons_L2_vis = neuron_detection_dict_vis["activated_neurons_L2"]
    shared_neurons_vis = neuron_detection_dict_vis["shared_neurons"]
    specific_neurons_L1_vis = neuron_detection_dict_vis["specific_neurons_L1"]
    specific_neurons_L2_vis = neuron_detection_dict_vis["specific_neurons_L2"]

    # shared_neuronsの各layer_idx, neuron_idxの発火値の合計 <- act_sum_shared_dictに保持

    """ for base line """
    # for visualization
    activated_neurons_L1_base_vis = neuron_detection_base_dict_vis["activated_neurons_L1"]
    activated_neurons_L2_base_vis = neuron_detection_base_dict_vis["activated_neurons_L2"]
    shared_neurons_base_vis = neuron_detection_base_dict_vis["shared_neurons"]
    specific_neurons_L1_base_vis = neuron_detection_base_dict_vis["specific_neurons_L1"]
    specific_neurons_L2_base_vis = neuron_detection_base_dict_vis["specific_neurons_L2"]

    """ 発火頻度dict """
    # freq_L1 = freq_dict["activated_neurons_L1"]
    # freq_L2 = freq_dict["activated_neurons_L2"]
    # freq_shared = freq_dict["shared_neurons"]
    # freq_L1_only = freq_dict["specific_neurons_L1"]
    # freq_L2_only = freq_dict["specific_neurons_L2"]

    """ 発火値の合計dict """
    # sum_shared = act_sum_dict["shared"]
    # sum_L1_or_L2 = act_sum_dict["L1_or_L2"]
    # sum_L1_specific = act_sum_dict["L1_specific"]
    # sum_L1_specific = act_sum_dict["L2_specific"]

    """ (初回だけ)pickleでfileにshared_neurons(track_dict)を保存
    freq dict: (2000対訳ペアを入れた時の） 各種ニューロンの発火頻度
    sum dict: 各種ニューロンの発火値の合計
    """
    # 対訳ペア(freq_dict)
    pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/pickles/act_freq/tatoeba_0_th/act_freq_dict_en_{L2}_tatoeba_0_th.pkl"
    # directoryを作成（存在しない場合のみ)
    os.makedirs(os.path.dirname(pkl_file_path), exist_ok=True)
    with open(pkl_file_path, "wb") as f:
        pickle.dump(freq_dict, f)
    print("pickle file saved.")

    # base (非対訳ペア)
    pkl_file_base_path = f"/home/s2410121/proj_LA/activated_neuron/pickles/base/act_freq/tatoeba_0_th/act_freq_dict_en_{L2}_tatoeba_0_th.pkl"
    # directoryを作成（存在しない場合のみ)
    os.makedirs(os.path.dirname(pkl_file_base_path), exist_ok=True)
    with open(pkl_file_base_path, "wb") as f:
        pickle.dump(freq_base_dict, f)
    print("pickle file saved.")

    """ pickle file(shared_neurons)の解凍/読み込み """
    # with open(pkl_file_path, "rb") as f:
    #     loaded_dict = pickle.load(f)
    # print("unfold pickle")
    # print(loaded_dict)
    # sys.exit()


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

    # visualize_neurons_with_line_plot_simple(
    #                                     L1,
    #                                     L2,
    #                                     # main
    #                                     activated_neurons_L1_vis,
    #                                     activated_neurons_L2_vis,
    #                                     non_activated_neurons_L1_vis,
    #                                     non_activated_neurons_L2_vis,
    #                                     shared_neurons_vis,
    #                                     specific_neurons_L1_vis,
    #                                     specific_neurons_L2_vis,
    #                                     non_activated_neurons_all_vis,
    #                                     "tatoeba_0.5_th",
    #                                     # base line
    #                                     shared_neurons_base_vis,
    #                                 )

    # visualize_neurons_with_line_plot(
    #                                     L1,
    #                                     L2,
    #                                     # main
    #                                     activated_neurons_L1,
    #                                     activated_neurons_L2,
    #                                     non_activated_neurons_L1,
    #                                     non_activated_neurons_L2,
    #                                     shared_neurons,
    #                                     specific_neurons_L1,
    #                                     specific_neurons_L2,
    #                                     non_activated_neurons_all,
    #                                     "tatoeba",
    #                                     # base line
    #                                     shared_neurons_base,
    #                                 )

    # visualize_neurons_with_line_plot_b(
    #                                     L1,
    #                                     L2,
    #                                     # base line
    #                                     activated_neurons_L1_base_vis,
    #                                     activated_neurons_L2_base_vis,
    #                                     non_activated_neurons_L1_base_vis,
    #                                     non_activated_neurons_L2_base_vis,
    #                                     shared_neurons_base_vis,
    #                                     specific_neurons_L1_base_vis,
    #                                     specific_neurons_L2_base_vis,
    #                                     non_activated_neurons_all_base_vis,
    #                                     "only_1s"
    #                                 )

if __name__ == "__main__":
    print('visualization completed.')
