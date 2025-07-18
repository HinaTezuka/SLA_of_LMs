import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from collections import defaultdict
from itertools import permutations

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import svd, matrix_rank
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from funcs import (
    unfreeze_pickle,
    save_as_pickle,
)

# langs = ["ja", "nl", "ko", "it", "en", "vi", "ru", "fr"]
langs = ["ja", "nl", "ko", "it", "en"]
# LLaMA3-8B / Mistral-7B / Aya-expanse-8B / BLOOM-3B.
model_names = ['CohereForAI/aya-expanse-8b', 'meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'bigscience/bloom-3b']
model_names = ['CohereForAI/aya-expanse-8b']
is_using_centroids = False
intervention_type = 'type-1'
# intervention_type = 'normal'
# intervention_type = 'ft' # fine-tuned.

""" compute distance between language subspaces. """
for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya' if 'aya' in model_name else 'bloom'
    layer_num = 33 if model_type in ['llama3', 'mistral', 'aya'] else 31 # emb_layer included.

    if intervention_type == 'normal':
        hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ja.pkl")
        hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/nl.pkl")
        hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ko.pkl")
        hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/it.pkl")
        hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/en.pkl")
        hs_vi = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/vi.pkl")
        hs_ru = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ru.pkl")
        hs_fr = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/fr.pkl")
    elif intervention_type == 'type-1':
        # normal
        hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/en_only_ja_type1.pkl")
        hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/nl_type1.pkl")
        hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ko_type1.pkl")
        hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/it_type1.pkl")
        hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/en_type1.pkl")
        # hs_vi = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/vi_type1.pkl")
        # hs_ru = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ru_type1.pkl")
        # hs_fr = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/fr_type1.pkl")
    elif intervention_type == 'type-2':
        hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/ja.pkl")
        hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/nl.pkl")
        hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/ko.pkl")
        hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/it.pkl")
        hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/en.pkl")
        hs_vi = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/vi.pkl")
        hs_ru = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/ru.pkl")
        hs_fr = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/fr.pkl")
    elif intervention_type == 'ft':
        # hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/ja.pkl")
        # hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/nl.pkl")
        # hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/ko.pkl")
        # hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/it.pkl")
        # hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/en.pkl")
        # hs_vi = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/vi.pkl")
        # hs_ru = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/ru.pkl")
        # hs_fr = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/fr.pkl")
        hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/ja_baseline.pkl")
        hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/nl_baseline.pkl")
        hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/ko_baseline.pkl")
        hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/it_baseline.pkl")
        hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/en_baseline.pkl")
        hs_vi = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/vi_baseline.pkl")
        hs_ru = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/ru_baseline.pkl")
        hs_fr = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ft/fr_baseline.pkl")

    sim_dict = sim_dict = defaultdict(lambda: defaultdict(lambda: np.float64(0)))

    for layer_i in range(layer_num):
        if (intervention_type == 'type-1' and layer_i in [ _ for _ in range(21, layer_num)]) or (intervention_type == 'type-2' and layer_i in [ _ for _ in range(21)]):
            continue

        hs_ja_layer = np.array(hs_ja[layer_i]) # shape: (n * d) n: sample_num, d: dimention of hs.
        hs_nl_layer = np.array(hs_nl[layer_i])
        hs_ko_layer = np.array(hs_ko[layer_i])
        hs_it_layer = np.array(hs_it[layer_i])
        hs_en_layer = np.array(hs_en[layer_i])
        # hs_vi_layer = np.array(hs_vi[layer_i])
        # hs_ru_layer = np.array(hs_ru[layer_i])
        # hs_fr_layer = np.array(hs_fr[layer_i])
    
        # compute cosine_sim beween vectors in each subspace.
        lang2hs_layer = {
            "ja": hs_ja_layer,
            "nl": hs_nl_layer,
            "ko": hs_ko_layer,
            "it": hs_it_layer,
            "en": hs_en_layer,
            # "vi": hs_vi_layer,
            # "ru": hs_ru_layer,
            # "fr": hs_fr_layer,
        }

        for lang1, lang2 in permutations(langs, 2):
            c1 = np.mean(lang2hs_layer[lang1], axis=0).reshape(1, -1)  # shape: (4096, )
            c2 = np.mean(lang2hs_layer[lang2], axis=0).reshape(1, -1)  # shape: (4096, )

            sim_score = cosine_similarity(c1, c2).item()

            # record all avg_sims into sim_dict
            sim_dict[f'{lang1}-{lang2}'][layer_i] = sim_score # lang1-lang2: average_sim between: i-th row vector in mat1(lang1) - row vectors in mat2(lang2)

        """ plot """
        lang_sim_matrix = np.zeros((len(langs), len(langs)))
        for i, lang1 in enumerate(langs):
            for j, lang2 in enumerate(langs):
                if lang1 == lang2:
                    lang_sim_matrix[i, j] = 1.0
                else:
                    key = f"{lang1}-{lang2}"
                    key_rev = f"{lang2}-{lang1}"

                    if key in sim_dict and layer_i in sim_dict[key]:
                        sim = sim_dict[key][layer_i]
                        lang_sim_matrix[i, j] = sim
                    elif key_rev in sim_dict and layer_i in sim_dict[key_rev]:
                        sim = sim_dict[key_rev][layer_i]
                        lang_sim_matrix[i, j] = sim
                    else:
                        lang_sim_matrix[i, j] = np.nan

        # Plot heatmap
        plt.rcParams["font.family"] = "DejaVu Serif"
        plt.figure(figsize=(8.5, 8))
        ax = sns.heatmap(
            lang_sim_matrix,
            xticklabels=langs,
            yticklabels=langs,
            annot=True,
            cmap="Blues",
            fmt=".2f",
            vmin=0,
            vmax=1,
            square=True,
            annot_kws={"size": 20}
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        title = 'LLaMA3-8B' if model_type == 'llama3' else 'Mistral-7B' if model_type == 'mistral' else 'Aya expanse-8B' if model_type == 'aya' else 'BLOOM-3B'
        plt.title(f"{title} - Layer {layer_i}", fontsize=30)
        plt.tick_params(labelsize=30)
        if intervention_type == 'type-1':
            for label in ax.get_xticklabels():
                if label.get_text() == 'en':
                    label.set_color('red')
                    label.set_fontweight('bold')
                    label.set_fontsize(40)

            for label in ax.get_yticklabels():
                if label.get_text() == 'en':
                    label.set_color('red')
                    label.set_fontweight('bold')
                    label.set_fontsize(40)

        if intervention_type == 'normal':
            save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/centroids"
            # save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/centroids/all"
        elif intervention_type == 'type-1':
            save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/type-1/centroids"
            save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/type-1/centroids/en_only_ja"
            # save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/type-1/centroids/all"
            # save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/type-1/centroids/qa"
        elif intervention_type == 'type-2':
            save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/type-2/centroids"
            # save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/type-2/centroids/all"
            # save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/type-2/centroids/all/n5000"
        elif intervention_type == 'ft':
            # save_dir = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/ft/centroids'
            save_dir = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/ft/centroids/baseline'
        os.makedirs(save_dir, exist_ok=True)
        if layer_i == 0:
            save_path = f'{save_dir}/emb_layer'
        else:
            save_path = f"{save_dir}/layer_{layer_i}"

        with PdfPages(save_path + '.pdf') as pdf:
            pdf.savefig(bbox_inches='tight', pad_inches=0.01)
            plt.close()