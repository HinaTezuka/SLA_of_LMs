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
)

langs = ["ja", "nl", "ko", "it", "en"]
# LLaMA3-8B / Mistral-7B / Aya-expanse-8B.
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", 'CohereForAI/aya-expanse-8b']
threshold_log = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
is_scaled = False

for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    layer_num = 41 if model_type == 'phi4' else 33 # emb_layer included.

    for layer_i in range(layer_num):
        all_lang_cumexp = {}
        all_lang_thresh = {}

        for L2 in langs:
            hs = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/{L2}.pkl")
            hs_layer = np.array(hs[layer_i]) # shape: (sample_num, hs_dim)
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
                print(f"{L2} - Layer {layer_i}: {int(t*100)}% variance explained by top {k} components")
                threshold_log[model_type][t][L2].append(k)

            all_lang_cumexp[L2] = cumulative_explained_variance
            all_lang_thresh[L2] = threshold_points

        plt.rcParams["font.family"] = "DejaVu Serif"
        plt.figure(figsize=(7, 6))

        colors_by_lang = {
            "en": "#1f77b4",
            "ja": "#ff7f0e",
            "ko": "#2ca02c",
            "it": "#d62728",
            "nl": "#9467bd"
        }

        y_offset_base = 0.9
        y_step = 0.05  # Vertical spacing between text annotations

        for i, lang in enumerate(langs):
            plt.plot(all_lang_cumexp[lang], color=colors_by_lang[lang], linewidth=3, label=lang)

            # Only annotate 95% threshold
            k95 = all_lang_thresh[lang][0.95]
            y_offset = y_offset_base - i * y_step  # Decrease y per language
            plt.axvline(x=k95, color=colors_by_lang[lang], linestyle="--", linewidth=1.5, alpha=0.7)
            plt.text(k95 + 5, y_offset, f"{lang} - 95% : {k95} components",
                    fontsize=18, fontweight="bold", color=colors_by_lang[lang])

        plt.axhline(y=0.95, color="#54AFE4", linestyle="--", linewidth=2)

        plt.xlabel("# Components", fontsize=30)
        plt.ylabel("Explained Variance", fontsize=30)
        plt.title(f"Layer {layer_i}", fontsize=30)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=14)

        if layer_i == 0:
            if is_scaled:
                path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/layer_wise/scale/emb_layer'
            else:
                path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/layer_wise/emb_layer'
        else:
            if is_scaled:
                path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/layer_wise/scale/{layer_i}'
            else:
                path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/layer_wise/{layer_i}'
        with PdfPages(path + '.pdf') as pdf:
            pdf.savefig(bbox_inches='tight', pad_inches=0.01)
            plt.close()

    # Summary plot
    if is_scaled:
        output_dir = "/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/summary/scale"
    else:
        output_dir = "/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/summary"
    os.makedirs(output_dir, exist_ok=True)

    colors_by_lang = {
        "en": "#1f77b4",
        "ja": "#ff7f0e",
        "ko": "#2ca02c",
        "it": "#d62728",
        "nl": "#9467bd"
    }

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
        for threshold in [0.9, 0.95, 0.99]:
            plt.figure(figsize=(10, 6))

            for lang in langs:
                y = threshold_log[model_type][threshold][lang]
                plt.plot(range(len(y)), y, label=lang, color=colors_by_lang[lang], linewidth=2)

            plt.title(f"{model_name_map[model_type]} - {int(threshold * 100)}% Variance", fontsize=30)
            plt.xlabel("Layers", fontsize=30)
            plt.ylabel("# Components", fontsize=30)
            plt.grid(True, linestyle=":", alpha=0.5)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(title="Language", fontsize=25, title_fontsize=25)
            plt.tight_layout()

            save_path = os.path.join(output_dir, f"{model_type}_{int(threshold * 100)}")
            with PdfPages(save_path + '.pdf') as pdf:
                pdf.savefig(bbox_inches='tight', pad_inches=0.01)
                plt.close()