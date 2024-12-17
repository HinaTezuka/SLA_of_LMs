import os
import itertools
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/hidden_state_sim/AUC")
import math
import dill as pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import average_precision_score

from expertise_funcs import (
    track_neurons_with_text_data,
    save_as_pickle,
    unfreeze_pickle,
)

# parameters
active_THRESHOLD = 0.01
L2 = "ja"

# unfreeze activation_dicts
pkl_path_same_semantics = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/same_semantics/{active_THRESHOLD}_th/en_{L2}.pkl"
pkl_path_non_same_semantics = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/non_same_semantics/{active_THRESHOLD}_th/en_{L2}.pkl"

# if os.path.exists(pkl_path_same_semantics):
#     os.remove(pkl_path_same_semantics)  # 破損ファイルを削除
# if os.path.exists(pkl_path_non_same_semantics):
#     os.remove(pkl_path_non_same_semantics)  # 破損ファイルを削除
# sys.exit()
act_same_semantics_dict = unfreeze_pickle(pkl_path_same_semantics)
# print(act_same_semantics_dict)
act_non_same_semantics_dict = unfreeze_pickle(pkl_path_non_same_semantics)
# print(act_same_semantics_dict)
# print(act_non_same_semantics_dict)
# sys.exit()
""" AUC calculation setting """
# def compute_ap_and_sort(label1_dict, label2_dict):
#     # 各ニューロンごとの活性化値とラベルを収集
#     neuron_responses = defaultdict(list)  # { (layer_idx, neuron_idx): [activation_values, ...] }
#     neuron_labels = defaultdict(list)     # { (layer_idx, neuron_idx): [labels, ...] }

#     # ラベル1の文を処理
#     for sentence_idx, layer_data in label1_dict.items():
#         for layer_idx, neuron_activations in layer_data.items():
#             for neuron_idx, activation_value in neuron_activations:
#                 neuron_responses[(layer_idx, neuron_idx)].append(activation_value)
#                 neuron_labels[(layer_idx, neuron_idx)].append(1)  # ラベル1: 正例

#     # ラベル2の文を処理
#     for sentence_idx, layer_data in label2_dict.items():
#         for layer_idx, neuron_activations in layer_data.items():
#             for neuron_idx, activation_value in neuron_activations:
#                 neuron_responses[(layer_idx, neuron_idx)].append(activation_value)
#                 neuron_labels[(layer_idx, neuron_idx)].append(0)  # ラベル2: 負例

#     # 各ニューロンのAPを計算
#     ap_scores = {}
#     for (layer_idx, neuron_idx), activations in neuron_responses.items():
#         labels = neuron_labels[(layer_idx, neuron_idx)]
#         ap = average_precision_score(y_true=labels, y_score=activations)
#         ap_scores[(layer_idx, neuron_idx)] = ap

#     # APスコア順にソート
#     sorted_neurons = sorted(ap_scores.keys(), key=lambda x: ap_scores[x], reverse=True)

#     return sorted_neurons, ap_scores

def normalize_activations(activations):
    """
    Min-Max normalization.
    Args:
        activations: 活性化値のリスト
    Returns:
        正規化された活性化値のリスト
    """
    min_val = min(activations)
    max_val = max(activations)

    # 避けるべきゼロ分母問題を防ぐ
    if max_val - min_val == 0:
        return [0.0 for _ in activations]

    # Min-Max正規化適用
    return [(x - min_val) / (max_val - min_val) for x in activations]


def compute_ap_and_sort(label1_dict, label2_dict, weight_factor=1.0):
    """
    calc APscore and sort (considering nums_of_label).

    Args:
        label1_dict: activation_value for correct label(1)
        label2_dict: activation_value for incorrect label(0)
        weight_factor: APスコアとラベル数の統合時の重み調整パラメータ

    Returns:
        ソートされたニューロンリスト, ニューロンごとの統合スコア辞書
    """
    # 各ニューロンごとの活性化値とラベルを収集
    neuron_responses = defaultdict(list)  # { (layer_idx, neuron_idx): [activation_values, ...] }
    neuron_labels = defaultdict(list)     # { (layer_idx, neuron_idx): [labels, ...] }

    # pairs for label:1
    for sentence_idx, layer_data in label1_dict.items():
        for layer_idx, neuron_activations in layer_data.items():
            for neuron_idx, activation_value in neuron_activations:
                neuron_responses[(layer_idx, neuron_idx)].append(activation_value)
                neuron_labels[(layer_idx, neuron_idx)].append(1)  # ラベル1: 正例

    # pairs for label:0
    for sentence_idx, layer_data in label2_dict.items():
        for layer_idx, neuron_activations in layer_data.items():
            for neuron_idx, activation_value in neuron_activations:
                neuron_responses[(layer_idx, neuron_idx)].append(activation_value)
                neuron_labels[(layer_idx, neuron_idx)].append(0)  # ラベル2: 負例

    # calc AP score for each shared neuron and calc total score which consider nums of label
    final_scores = {}
    for (layer_idx, neuron_idx), activations in neuron_responses.items():
        labels = neuron_labels[(layer_idx, neuron_idx)]

        # normalization
        normalized_activations = normalize_activations(activations)

        # AP score
        ap = average_precision_score(y_true=labels, y_score=normalized_activations)

        label_count = len(labels) # nums of label(to be considered)
        # calc total score
        final_score = ap * math.log(1 + label_count) * weight_factor
        # save total score
        final_scores[(layer_idx, neuron_idx)] = final_score

    # sort: based on total score of each neuron
    sorted_neurons = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)

    return sorted_neurons, final_scores

""" """
sorted_neurons, ap_scores = compute_ap_and_sort(act_same_semantics_dict, act_non_same_semantics_dict)

# save as pickle file
sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/ap_scores/sorted_neurons_{L2}.pkl"
ap_scores_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/ap_scores/ap_scores_{L2}.pkl"
save_as_pickle(sorted_neurons_path, sorted_neurons)
save_as_pickle(ap_scores_path, ap_scores)

# unfreeze pickle
sorted_neurons = unfreeze_pickle(sorted_neurons_path)
ap_scores_path = unfreeze_pickle(ap_scores_path)

print(ap_scores)
print(sorted_neurons[:30])
# print(len(sorted_neurons))
