""" visualize magnitude of changes of params with heatmap for LLaMA3"""
import os
import sys
sys.path.append("/home/s2410121/proj_LA/measure_similarities/llama3")
import re

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from llama3_measure_similarity import *

# ネストされたdefaultdictを生成するためのヘルパー関数
def nested_defaultdict():
    return defaultdict(float)

# 二重ネスト構造のparam_changes_per_arc
# param_changes_per_arc = {
#     'attention': defaultdict(nested_defaultdict),
#     'mlp': defaultdict(nested_defaultdict),
#     'layernorm': defaultdict(nested_defaultdict),
# }
# param_changes_per_arc = {
#     'attention': {},
#     'mlp': {},
#     'layernorm': {},
# }

# get changes of weight per each langs <- only for abs changes of params(not for cos_sim and biases)
def get_param_changes(weight_changes, langs: str) -> dict:
    param_changes_per_arc = {
        'attention': {},
        'mlp': {},
        'layernorm': {},
    }
    for k, v in weight_changes[langs].items():
        # 数字の部分を抽出
        layer_num = re.search(r'layers\.(\d+)\.', k)
        if layer_num:
            layer_num = layer_num.group(1)  # 数字部分だけを取得
        # layers.29.などの層のidx名を消す
        modified_key = re.sub(r'layers\.\d+\.', '', k)
        if 'attn' in k and 'weight' in k:
            if modified_key not in param_changes_per_arc['attention']:
                param_changes_per_arc['attention'][modified_key] = []
                param_changes_per_arc['attention'][modified_key].append(v['mean_absolute_difference'])
            else:
                param_changes_per_arc['attention'][modified_key].append(v['mean_absolute_difference'])
        elif 'mlp' in k and 'weight' in k:
            if modified_key not in param_changes_per_arc['mlp']:
                param_changes_per_arc['mlp'][modified_key] = []
                param_changes_per_arc['mlp'][modified_key].append(v['mean_absolute_difference'])
            else:
                param_changes_per_arc['mlp'][modified_key].append(v['mean_absolute_difference'])
        elif 'layernorm' in k and 'weight' in k:
            if modified_key not in param_changes_per_arc['layernorm']:
                param_changes_per_arc['layernorm'][modified_key] = []
                param_changes_per_arc['layernorm'][modified_key].append(v['mean_absolute_difference'])
            else:
                param_changes_per_arc['layernorm'][modified_key].append(v['mean_absolute_difference'])

    return param_changes_per_arc

# Plotting function
def plot_heatmap(param_changes_per_arc, title="heatmap_gpt2", lang_pair="en_ja") -> None:
    data = {}

    # データ整形: 32層に対応するリストを作成
    for param_type, changes in param_changes_per_arc.items():
        for param_name, change_list in changes.items():
            # 32層のデータを作成（足りない層は0で補完）
            data[param_name] = (change_list + [0] * 32)[:32]

    # DataFrame作成（行: 層, 列: modified_key）
    df = pd.DataFrame(data, index=[f"{i}" for i in range(32)])

    # プロット
    plt.figure(figsize=(30, 15))
    sns.heatmap(df, annot=True, fmt=".4f", cmap='Greens', vmin=0.0001, vmax=0.0100, cbar_kws={'label': 'changes of weights'})
    plt.title(f'{title} {lang_pair}', fontsize=20, pad=15)
    plt.xlabel('param_names', fontsize=16, labelpad=15)
    plt.ylabel('layer_idx', fontsize=16, labelpad=15)
    plt.savefig(f'/home/s2410121/proj_LA/measure_similarities/llama3/images/{title}_{lang_pair}.png')
    plt.close()

# def plot_heatmap(param_changes_per_arc, title="heatmap_gpt2", lang_pair="en_ja") -> None:
#     data = defaultdict(list)
#     for v in param_changes_per_arc.values():
#         for param_name in v.keys():
#             data[param_name] = v[param_name]
#     # layer_idx (vertical)
#     # layer_idx = list(range(len(next(iter(data.values())))))
#     layer_idx = list(range(32))
#     # layer_idx = layer_idx[::-1]
#     # df creation
#     df = pd.DataFrame(data)
#     # plot
#     plt.figure(figsize=(30, 15))
#     sns.heatmap(df, annot=True, cmap='Greens', vmin=0.0001, vmax=0.0025, cbar_kws={'label': 'changes of weights'})
#     plt.title(f'{title}{lang_pair}')
#     # plt.xlabel('param_names')
#     plt.ylabel('layer_idx')
#     plt.savefig(f'/home/s2410121/proj_LA/measure_similarities/llama3/images/{title}{lang_pair}.png')
#     plt.close()


if __name__ == "__main__":

    """ llama3 """
    # en_ja
    param_changes_per_arc = get_param_changes(weight_changes_llama, 'en_ja')
    # print(param_changes_per_arc)
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_ja")

    # en_ger
    param_changes_per_arc = get_param_changes(weight_changes_llama, 'en_ger')
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_ger")

    # # en_ita
    param_changes_per_arc = get_param_changes(weight_changes_llama, 'en_ita')
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_ita")

    # # en_ko
    param_changes_per_arc = get_param_changes(weight_changes_llama, 'en_ko')
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_ko")

    print('visualization completed!')
