import os
import sys
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from funcs import unfreeze_pickle

langs = ['ja', 'nl', 'ko', 'it']
model_types = ['llama3', 'mistral', 'aya']
score_types = ['cos_sim']
top_n = 1000

# 比較する言語ペア
lang_pairs = [('ja', 'nl'), ('ja', 'ko'), ('ja', 'it'), ('nl', 'ko'), ('nl', 'it'), ('ko', 'it')]

for model_type in model_types:
    layer_range = range(20)
    for score_type in score_types:
        plt.rcParams["font.family"] = "DejaVu Serif"
        plt.figure(figsize=(10, 10))

        # 各言語ごとに layer -> [neuron indices] の dict を作る
        lang_neuron_by_layer = dict()

        for lang in langs:
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{lang}_sorted_neurons.pkl"
            sorted_neurons = unfreeze_pickle(path)
            sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in layer_range][:top_n]

            layer_dict = defaultdict(list)
            for layer, neuron in sorted_neurons:
                layer_dict[layer].append(neuron)
            lang_neuron_by_layer[lang] = layer_dict

        # 言語ペアごとに Jaccard Index を layer ごとに計算
        for lang1, lang2 in lang_pairs:
            jaccard_per_layer = []
            for layer in layer_range:
                set1 = set(lang_neuron_by_layer[lang1].get(layer, []))
                set2 = set(lang_neuron_by_layer[lang2].get(layer, []))
                if not set1 and not set2:
                    jaccard = 0.0
                else:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                jaccard_per_layer.append(jaccard)

            plt.plot(list(layer_range), jaccard_per_layer, label=f"{lang1}-{lang2}")

        model_title = 'LLaMA3-8B' if model_type == 'llama3' else 'Mistral-7B' if model_type == 'mistral' else 'Aya expanse-8B'
        # プロット設定
        plt.xlabel("Layer Index", fontsize=35)
        plt.ylabel("Jaccard Index", fontsize=35)
        plt.title(f"{model_title}", fontsize=40)
        plt.tick_params(axis='x', labelsize=25)
        plt.tick_params(axis='y', labelsize=25)
        plt.legend(fontsize=25)
        plt.grid(True)
        plt.tight_layout()

        save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/jaccard/{model_type}_{score_type}"
        with PdfPages(save_path + '.pdf') as pdf:
            pdf.savefig(bbox_inches='tight', pad_inches=0.01)
            plt.close()