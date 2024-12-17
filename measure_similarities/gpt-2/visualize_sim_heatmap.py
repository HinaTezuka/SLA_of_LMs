""" visualize magnitude of changes of params with heatmap """
import os
import sys
sys.path.append("/home/s2410121/proj_LA/measure_similarities/gpt2/")
import re

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gpt2_measure_similarity import *

# # Define architectures and layer indices
# param_changes_per_arc = {
#                             'attention': {},
#                             'mlp': {},
#                             'ln': {},
#                         }

# get changes of weight per each langs <- only for abs changes of params(not for cos_sim and biases)
def get_param_changes(weight_changes, langs: str) -> dict:
    # Define architectures and layer indices
    param_changes_per_arc = {
                                'attention': {},
                                'mlp': {},
                                'ln': {},
                            }
    # print(weight_changes)
    for k, v in weight_changes[langs].items():
        # h.[].などの層のidx名を消す
        modified_key = re.sub(r'h\.\d+\.', '', k)
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
        elif 'ln' in k and 'weight' in k and 'ln_f' not in k:
            if modified_key not in param_changes_per_arc['ln']:
                param_changes_per_arc['ln'][modified_key] = []
                param_changes_per_arc['ln'][modified_key].append(v['mean_absolute_difference'])
            else:
                param_changes_per_arc['ln'][modified_key].append(v['mean_absolute_difference'])

    return param_changes_per_arc

# Plotting function
def plot_heatmap(param_changes_per_arc, title="heatmap_gpt2", lang_pair="en_ja") -> None:
    data = {}

    # データ整形: 12層に対応するリストを作成
    for param_type, changes in param_changes_per_arc.items():
        for param_name, change_list in changes.items():
            # 12層のデータを作成（足りない層は0で補完）
            data[param_name] = (change_list + [0] * 12)[:12]

    # DataFrame作成（行: 層, 列: modified_key）
    df = pd.DataFrame(data, index=[f"{i}" for i in range(12)])

    # プロット
    plt.figure(figsize=(30, 15))
    sns.heatmap(df, annot=True, fmt=".4f", cmap='Greens', vmin=0.05, vmax=0.3, cbar_kws={'label': 'changes of weights'})
    plt.title(f'{title} {lang_pair}', fontsize=20, pad=15)
    plt.xlabel('param_names', fontsize=16, labelpad=15)
    plt.ylabel('layer_idx', fontsize=16, labelpad=15)
    plt.savefig(f'/home/s2410121/proj_LA/measure_similarities/gpt-2/images/heat_maps/{title}_{lang_pair}.png')
    plt.close()


if __name__ == "__main__":
    """ gpt2 """
    # en_ja
    # param_changes_per_arc = get_param_changes(weight_changes_gpt2, 'en_ja')
    # # print(param_changes_per_arc)
    # plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_ja")

    # en_du
    param_changes_per_arc = get_param_changes(weight_changes_gpt2, 'en_du')
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_du")

    # en_ger
    # param_changes_per_arc = get_param_changes(weight_changes_gpt2, 'en_ger')
    # plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_ger")

    # en_ita
    param_changes_per_arc = get_param_changes(weight_changes_gpt2, 'en_ita')
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_ita")

    # en_fre
    param_changes_per_arc = get_param_changes(weight_changes_gpt2, 'en_fre')
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_fre")

    # en_ko
    # param_changes_per_arc = get_param_changes(weight_changes_gpt2, 'en_ko')
    # plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_ko")

    print('visualization completed!')
