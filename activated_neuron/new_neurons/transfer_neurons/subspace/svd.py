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

from funcs import (
    unfreeze_pickle,
)

langs = ["ja", "nl", "ko", "it", "en"]
# LLaMA3-8B / Mistral-7B / Aya-expanse-8B.
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", 'CohereForAI/aya-expanse-8b']
threshold_log = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # ← これを追加

for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    layer_num = 41 if model_type == 'phi4' else 33 # emb_layer included.

    for L2 in langs:
        hs = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/{L2}.pkl")
        for layer_i in range(layer_num):
            hs_layer = np.array(hs[layer_i]) # shape: (sample_num, hs_dim) <- ex. llama: np.array((1000, 4096))
            # singular value decomposition
            u, s, vh = svd(hs_layer, full_matrices=False) # s: list of singular values.

            # test (whether SVD correctly works)
            # print(np.allclose(u @ np.diag(s) @ vh, hs_layer, atol=1e-6)) # True.

            """ compute 累積寄与率（cumulative explained variance ratio) and plot. """
            # # Compute explained variance ratio
            # explained_variance_ratio = (s ** 2) / np.sum(s ** 2)

            # # Compute cumulative explained variance
            # cumulative_explained_variance = np.cumsum(explained_variance_ratio)

            # # Set thresholds for saturation levels
            # thresholds = [0.9, 0.95, 0.99]

            # # Find the number of components needed to reach each threshold
            # threshold_points = {}
            # for t in thresholds:
            #     k = np.searchsorted(cumulative_explained_variance, t) + 1  # +1 for human-readable index
            #     threshold_points[t] = k
            #     print(f"{L2} - Layer {layer_i}: {int(t*100)}% variance explained by top {k} components")

            # # Plot
            # plt.rcParams["font.family"] = "DejaVu Serif"
            # plt.figure(figsize=(7, 6))
            # plt.plot(cumulative_explained_variance, color="blue", linewidth=3)

            # colors = {
            #     0.9: "#08c93b",   # light blue
            #     0.95: "#54AFE4",  # medium blue
            #     0.99: "#24158A",  # darker blue
            # }
            # for t, k in threshold_points.items():
            #     plt.axhline(y=t, color=colors[t], linestyle="--", linewidth=2)
            #     plt.axvline(x=k, color=colors[t], linestyle="--", linewidth=2)
            #     plt.text(k + 5, t - 0.05, f"{int(t*100)}% : {k} components", fontsize=18, fontweight="bold", color=colors[t])

            # plt.xlabel("# Components", fontsize=30)
            # plt.ylabel("Explained Variance", fontsize=30)
            # plt.title(f"{L2}, Layer {layer_i}", fontsize=30)
            # plt.tick_params(axis='both', which='major', labelsize=20)
            # plt.grid(True, linestyle=":", alpha=0.6)

            # if layer_i == 0:
            #     path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/layer_wise/emb_layer'
            # else:
            #     path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/layer_wise/{layer_i}'
            # with PdfPages(path + '.pdf') as pdf:
            #     pdf.savefig(bbox_inches='tight', pad_inches=0.01)
            #     plt.close()

            """  """
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
                0.9: "#08c93b",   # light blue
                0.95: "#54AFE4",  # medium blue
                0.99: "#24158A",  # darker blue
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

                    plt.title(f"{model_name_map[model_type]} - {int(threshold * 100)}% Variance", fontsize=20)
                    plt.xlabel("Layer", fontsize=18)
                    plt.ylabel("# Components", fontsize=18)
                    plt.grid(True, linestyle=":", alpha=0.5)
                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.legend(title="Language", fontsize=14, title_fontsize=14)
                    plt.tight_layout()

                    save_path = os.path.join(output_dir, f"{model_type}_{int(threshold * 100)}.pdf")
                    with PdfPages(save_path + '.pdf') as pdf:
                        pdf.savefig(bbox_inches='tight', pad_inches=0.01)
                        plt.close()
