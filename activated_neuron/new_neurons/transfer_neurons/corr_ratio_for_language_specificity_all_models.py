import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from funcs import (
    multilingual_dataset_for_lang_specific_detection,
    track_neurons_with_text_data,
    save_as_pickle,
    compute_ap_and_sort,
    unfreeze_pickle,
    save_np_arrays,
    unfreeze_np_arrays,
)

# langs = ['ja', 'nl', 'ko', 'it']
# langs = ['nl', 'it']
# langs = ['ja', 'ko']
langs = ['fr']
model_types = ['llama3', 'mistral', 'aya']
# score_types = ['cos_sim', 'L2_dis']
score_types = ['cos_sim']
is_reverses = [False, True]

def correlationRatio(categories, values):
    interclass_variation  = sum([
        (len(values[categories == i]) * ((values[categories == i].mean() - values.mean()) ** 2)).sum() for i in np.unique(categories)
    ]) 
    total_variation = sum((values - values.mean()) ** 2)
    return interclass_variation / total_variation

l1 = [ 1 for _ in range(1000)]
l2 = [ 0 for _ in range(1000)]

# # 5言語用
# labels_dict = {
#     'ja': l1 + l2 + l2 + l2 + l2,
#     'nl': l2 + l1 + l2 + l2 + l2,
#     'ko': l2 + l2 + l1 + l2 + l2,
#     'it': l2 + l2 + l2 + l1 + l2,
#     'en': l2 + l2 + l2 + l2 + l1,
# }
# 全言語
labels_dict = {
    'ja': l1 + l2 + l2 + l2 + l2 + l2 + l2 + l2,
    'nl': l2 + l1 + l2 + l2 + l2 + l2 + l2 + l2,
    'ko': l2 + l2 + l1 + l2 + l2 + l2 + l2 + l2,
    'it': l2 + l2 + l2 + l1 + l2 + l2 + l2 + l2,
    'en': l2 + l2 + l2 + l2 + l1 + l2 + l2 + l2,
    # 'vi': l2 + l2 + l2 + l2 + l2 + l1 + l2 + l2,
    # 'ru': l2 + l2 + l2 + l2 + l2 + l2 + l1 + l2,
    'fr': l2 + l2 + l2 + l2 + l2 + l2 + l2 + l1,
}
    
""" normalな計算用 """
for model_type in model_types:
    for L2 in langs:
        for is_reverse in is_reverses:
            for score_type in score_types:
                # activations
                save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/activations/{L2}_last_token.npz"
                # save_path_labels = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/labels/{L2}_last_token.pkl"
                activations_arr = unfreeze_np_arrays(save_path_activations)
                # labels_list = np.array(unfreeze_pickle(save_path_labels))

                """ calc corr_ratio. """
                # top score neurons
                if is_reverse:
                    save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
                    sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
                    sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]] if model_type in ['llama3', 'mistral', 'aya'] else [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20, 30)]]
                elif not is_reverse:
                    save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
                    sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
                    sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20)]]

                # prepare labels
                labels_list = labels_dict[L2]

                top_n = 1000
                corr_ratios = defaultdict(float)
                arr = []
                for (layer_i, neuron_i) in sorted_neurons[:top_n]:
                    corr_ratio = correlationRatio(labels_list, activations_arr[layer_i, neuron_i, :])
                    corr_ratios[(layer_i, neuron_i)] = corr_ratio
                    arr.append(corr_ratio)

                neuron_type = 'type-2' if is_reverse else 'type-1'
                print(f'{model_type}, {neuron_type}, {L2}, {score_type}')
                print(np.mean(np.array(arr)))


# """ 可視化用 """
# for model_type in model_types:
#     for is_reverse in is_reverses:
#         for score_type in score_types:

#             # データ格納用
#             plot_data = []

#             for L2 in langs:
#                 # activations
#                 save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/activations/{L2}_last_token.npz"
#                 activations_arr = unfreeze_np_arrays(save_path_activations)

#                 if is_reverse:
#                     save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
#                     sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
#                     sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in range(20, 30)] if model_type == 'bloom' else [neuron for neuron in sorted_neurons if neuron[0] in range(20, 32)]
#                 else:
#                     save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
#                     sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
#                     sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in range(20)]

#                 labels_list = labels_dict[L2]

#                 top_n = 1000
#                 arr = []
#                 for (layer_i, neuron_i) in sorted_neurons[:top_n]:
#                     corr_ratio = correlationRatio(labels_list, activations_arr[layer_i, neuron_i, :])
#                     arr.append(corr_ratio)

#                 for rank, score in enumerate(arr):
#                     plot_data.append({
#                         'Rank': rank,
#                         'Correlation Ratio': score,
#                         'Language': L2
#                     })

#                 neuron_type = 'type-2' if is_reverse else 'type-1'
#                 print(f'{model_type}, {neuron_type}, {L2}, {score_type}')
#                 print(f"mean corr_ratio: {np.mean(arr):.4f}")

#             # ======== 📈 可視化（Rank vs Corr Ratioの滑らかな分布＋logスケール） ========
#             df_plot = pd.DataFrame(plot_data)

#             plt.figure(figsize=(10, 6))

#             # 各言語ごとに曲線を描く
#             for lang in df_plot["Language"].unique():
#                 df_lang = df_plot[df_plot["Language"] == lang]

#                 # 並び順保証
#                 df_lang = df_lang.sort_values(by="Rank")

#                 # 任意で平滑化：rolling mean (任意で使ってもOK)
#                 # df_lang["Correlation Ratio"] = df_lang["Correlation Ratio"].rolling(window=10, min_periods=1).mean()

#                 sns.lineplot(
#                     x=df_lang["Rank"] + 1,  # rankは0始まりなので+1
#                     y=df_lang["Correlation Ratio"],
#                     label=lang,
#                     linewidth=1.5
#                 )

#             model_title = {
#                 'llama3': 'LLaMA3-8B',
#                 'mistral': 'Mistral-7B',
#                 'aya': 'Aya Expanse-8B',
#                 'bloom': 'BLOOM'
#             }.get(model_type, model_type)

#             neuron_type = 'type-2' if is_reverse else 'type-1'
#             plt.title(f"{model_title}, Type: {neuron_type}, Score: {score_type}")
#             plt.xlabel("Neuron Rank (Sorted)")
#             plt.ylabel("Correlation Ratio Score (η²)")
#             plt.xscale('log')  # 横軸をlogスケールに
#             plt.grid(True, which='both', linestyle='--', alpha=0.6)
#             plt.legend(title="Language")
#             plt.tight_layout()

#             # 保存
#             path_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/corr_ratio/{neuron_type}"
#             os.makedirs(path_dir, exist_ok=True)
#             path = os.path.join(path_dir, f"{model_type}_{score_type}.pdf")

#             with PdfPages(path) as pdf:
#                 pdf.savefig(bbox_inches='tight', pad_inches=0.01)
#                 plt.close()