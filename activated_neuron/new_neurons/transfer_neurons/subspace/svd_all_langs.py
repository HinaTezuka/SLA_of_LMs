import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.linalg import svd, matrix_rank
from matplotlib.backends.backend_pdf import PdfPages
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler

from funcs import (
    unfreeze_pickle,
    save_as_pickle,
)

langs = ["ja", "nl", "ko", "it", "en"]
# LLaMA3-8B / Mistral-7B / Aya-expanse-8B.
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", 'CohereForAI/aya-expanse-8b']
threshold_log = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
is_scaled = False

for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    layer_num = 41 if model_type == 'phi4' else 33 # emb_layer included.
    # load hidden states.
    hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ja.pkl")
    hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/nl.pkl")
    hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ko.pkl")
    hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/it.pkl")
    hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/en.pkl")

    for layer_i in range(layer_num):
        all_lang_cumexp = {}
        all_lang_thresh = {}

        hs_ja_layer = np.array(hs_ja[layer_i]) # shape: (n * d) n: sample_num, d: dimention of hs.
        hs_nl_layer = np.array(hs_nl[layer_i])
        hs_ko_layer = np.array(hs_ko[layer_i])
        hs_it_layer = np.array(hs_it[layer_i])
        hs_en_layer = np.array(hs_en[layer_i])
        hs_layer = np.concatenate([hs_ja_layer, hs_nl_layer, hs_ko_layer, hs_it_layer, hs_en_layer], axis=0)

        if is_scaled:
            scaler = StandardScaler()
            hs_layer = scaler.fit_transform(hs_layer)
        u, s, vh = svd(hs_layer, full_matrices=False)

        explained_variance_ratio = (s ** 2) / np.sum(s ** 2) # 寄与率.
        cumulative_explained_variance = np.cumsum(explained_variance_ratio) # 累積寄与率.
        thresholds = [0.9, 0.95, 0.99]

        threshold_points = {}
        for t in thresholds:
            k = np.searchsorted(cumulative_explained_variance, t) + 1
            threshold_points[t] = k
            print(f"Layer {layer_i}: {int(t*100)}% variance explained by top {k} components")
            threshold_log[model_type][t]["all"].append(k)  # 言語ではなく "all" とするのが自然

        plt.rcParams["font.family"] = "DejaVu Serif"
        plt.figure(figsize=(7, 6))

        # 累積寄与率のプロット（全言語統合の1本のみ）
        plt.plot(cumulative_explained_variance, color="#1f77b4", linewidth=3, label="All languages")

        # 95%しきい値の線と注釈
        k95 = threshold_points[0.95]
        plt.axvline(x=k95, color="#1f77b4", linestyle="--", linewidth=1.5, alpha=0.7)
        plt.text(k95 + 5, 0.87, f"95% : {k95} components",
                fontsize=18, fontweight="bold", color="#1f77b4")

        plt.axhline(y=0.95, color="#54AFE4", linestyle="--", linewidth=2)

        plt.xlabel("# Components", fontsize=30)
        plt.ylabel("Explained Variance", fontsize=30)
        plt.title(f"Layer {layer_i}", fontsize=30)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=20)

        if layer_i == 0:
            if is_scaled:
                path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/layer_wise/all/scale/emb_layer'
            else:
                path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/layer_wise/all/emb_layer'
        else:
            if is_scaled:
                path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/layer_wise/all/scale/{layer_i}'
            else:
                path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/layer_wise/all/{layer_i}'
        with PdfPages(path + '.pdf') as pdf:
            pdf.savefig(bbox_inches='tight', pad_inches=0.01)
            plt.close()

    # Summary plot
    if is_scaled:
        output_dir = "/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/summary/all/scale"
    else:
        output_dir = "/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/summary/all"
    os.makedirs(output_dir, exist_ok=True)

    threshold_colors = {
        0.9: "#08c93b",
        0.95: "#54AFE4",
        0.99: "#24158A",
    }

    model_name_map = {
        "llama3": "LLaMA3-8B",
        "mistral": "Mistral-7B",
        "aya": "Aya-8B"
    }

    for model_type in threshold_log:
        plt.figure(figsize=(10, 6))

        for threshold in [0.9, 0.95, 0.99]:
            y = threshold_log[model_type][threshold]["all"]
            plt.plot(
                range(len(y)),
                y,
                color=threshold_colors[threshold],
                linewidth=3,
                label=f"{int(threshold * 100)}%"
            )

        plt.title(f"{model_name_map[model_type]} - Variance Thresholds", fontsize=30)
        plt.xlabel("Layers", fontsize=30)
        plt.ylabel("# Components", fontsize=30)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=22, title="Threshold", title_fontsize=20)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{model_type}_all_thresholds")
        with PdfPages(save_path + '.pdf') as pdf:
            pdf.savefig(bbox_inches='tight', pad_inches=0.01)
            plt.close()

# save threshold log as pkl.
path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/subspace/dist_between_subspaces/threshold_log_pca.pkl'
save_as_pickle(path, dict(threshold_log))