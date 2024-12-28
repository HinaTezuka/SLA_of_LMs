"""
visualize experts for same semantics as a sentence.
"""
import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/hidden_state_sim/AUC")
import dill as pickle
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from expertise_funcs import (
    save_as_pickle,
    unfreeze_pickle,
)


""" parameters setting """
# activation_type = "abs"
activation_type = "product"
norm_type = "no"
# norm_type = "min_max"
# norm_type = "sigmoid"
# L2 = "ja"
L2_list = ["ja", "nl", "ko", "it"]

for L2 in L2_list:

    """ unfreeze pickles sorted based on AP. """
    sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/sorted_neurons_{L2}.pkl"
    ap_scores_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/ap_scores_{L2}.pkl"
    # unfreeze pickle
    sorted_neurons = unfreeze_pickle(sorted_neurons_path)
    ap_scores = unfreeze_pickle(ap_scores_path)

    #
    sorted_neurons = sorted_neurons[:5000]
    print(sorted_neurons)
    # for i in sorted_neurons:
    #     print(ap_scores[i])
    # sys.exit()

    # データ準備
    max_layer = 32
    layers = [layer for layer, neuron in sorted_neurons]  # 層のみ抽出

    # 層ごとのカウント
    layer_counts = Counter(layers)

    # ヒストグラムプロット
    plt.figure(figsize=(12, 6))
    plt.bar(layer_counts.keys(), layer_counts.values(), color="skyblue", edgecolor=None)
    plt.xticks(range(max_layer), fontsize=10)
    plt.xlabel("Layer Index")
    plt.ylabel("Neurons Count")
    plt.title("Same Semantic Expert Shared Neurons Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 保存
    plt.savefig(
        f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/same_semantic_experts/{activation_type}/{L2}.png',
        bbox_inches='tight',
        )
    plt.close()
