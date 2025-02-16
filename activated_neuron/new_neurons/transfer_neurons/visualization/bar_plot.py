"""
visualize experts for same semantics as a sentence.
"""
import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons")
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    save_as_pickle,
    unfreeze_pickle,
)


""" parameters setting """
model = "llama"
# model = "gpt2"
activation_type = "abs"
# activation_type = "product"
norm_type = "no"
# norm_type = "min_max"
norm_type = "sigmoid"
# L2 = "ja"
L2_list = ["ja", "nl", "ko", "it"]
L2_list = ["ko"]

for L2 in L2_list:

    # L2-specific neurons
    save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/sorted_neurons_{L2}.pkl"
    save_path_ap_scores = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/ap_scores_{L2}.pkl"
    sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
    ap_scores = unfreeze_pickle(save_path_ap_scores)

    # #
    sorted_neurons = sorted_neurons[:1000] + sorted_neurons[-1000:]
    # # sorted_neurons = sorted_neurons
    # # print(sorted_neurons)
    
    # """ 上位10件を表示 """
    # print(f"======================== {L2} ========================")
    # for i in sorted_neurons[:10]:
    #     print(ap_scores[i])
    # print(f"=====================================================")
    # for i in sorted_neurons[-10:]:
    #     print(ap_scores[i])
    # sys.exit()
    # print(f"30000番目: {ap_scores[sorted_neurons[10000]]}")
    # # sys.exit()

    # データ準備
    max_layer = 32 if model == "llama" else 12
    layers = [layer for layer, neuron in sorted_neurons]  # 層のみ抽出

    # 層ごとのカウント
    layer_counts = Counter(layers)

    # ヒストグラムプロット
    plt.figure(figsize=(12, 6))
    plt.bar(layer_counts.keys(), layer_counts.values(), color="purple", edgecolor=None)
    plt.xticks(range(max_layer))
    plt.xlabel("Layer Index", fontsize=40)
    plt.ylabel("Neurons Count", fontsize=40)
    plt.title("Ja specific neuorns")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 保存
    plt.savefig(
        f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/{L2}.png',
        bbox_inches='tight',
        )
    # plt.close()
    sys.exit()
