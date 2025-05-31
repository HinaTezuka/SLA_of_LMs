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
    defaultdict_to_dict,
)

langs = ["ja", "nl", "ko", "it", "en"]
# LLaMA3-8B / Mistral-7B / Aya-expanse-8B.
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", 'CohereForAI/aya-expanse-8b']
is_using_centroids = False
intervention_type = 'normal'

""" compute distance between language subspaces. """
for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    layer_num = 41 if model_type == 'phi4' else 33 # emb_layer included.

    if intervention_type == 'normal':
        hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ja.pkl")
        hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/nl.pkl")
        hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ko.pkl")
        hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/it.pkl")
        hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/en.pkl")
    elif intervention_type == 'type-1':
        hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ja_type1.pkl")
        hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/nl_type1.pkl")
        hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ko_type1.pkl")
        hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/it_type1.pkl")
        hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/en_type1.pkl")
    elif intervention_type == 'type-2':
        hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/ja.pkl")
        hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/nl.pkl")
        hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/ko.pkl")
        hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/it.pkl")
        hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/en.pkl")

    sim_dict = defaultdict(lambda: defaultdict(list)) # {lang1-lang2: layer_idx: [cos_sim1, cos_sim2, ... , cos_sim1000} 1000: total_num_of_sample_sentences.

    for layer_i in range(layer_num):
        hs_ja_layer = np.array(hs_ja[layer_i]) # shape: (n * d) n: sample_num, d: dimention of hs.
        hs_nl_layer = np.array(hs_nl[layer_i])
        hs_ko_layer = np.array(hs_ko[layer_i])
        hs_it_layer = np.array(hs_it[layer_i])
        hs_en_layer = np.array(hs_en[layer_i])
        
        # compute cosine_sim beween vectors in each subspace.
        lang2hs_layer = {
            "ja": hs_ja_layer,
            "nl": hs_nl_layer,
            "ko": hs_ko_layer,
            "it": hs_it_layer,
            "en": hs_en_layer
        }

        for lang1, lang2 in permutations(langs, 2):
            mat1 = lang2hs_layer[lang1]  # shape: (1000, 4096)
            mat2 = lang2hs_layer[lang2]  # shape: (1000, 4096)

            sim_matrix = cosine_similarity(mat1, mat2)

            # average similarity for each row in mat1 against all rows in mat2
            avg_sims = np.mean(sim_matrix, axis=1)  # shape: (1000,)

            # record all avg_sims into sim_dict
            sim_dict[f'{lang1}-{lang2}'][layer_i].extend(avg_sims.tolist()) # lang1-lang2: average_sim between: i-th row vector in mat1(lang1) - row vectors in mat2(lang2)

    # save as pkl file.
    if intervention_type == 'normal':
        path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/subspace/dist_between_subspaces/{model_type}.pkl'
    elif intervention_type == 'type-1':
        path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/subspace/dist_between_subspaces/{model_type}_type1.pkl'
    elif intervention_type == 'type-2':
        path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/subspace/dist_between_subspaces/{model_type}_type2.pkl'
    save_as_pickle(path, defaultdict_to_dict(sim_dict))
    print(f'saving completed: {model_type}')

    """ plot """
    sim_dict = unfreeze_pickle(path)  # { 'lang1-lang2': {layer: [sim1, sim2, ...]}, ... }
    
    layer_num = 33 if model_type in ["llama3", "mistral", "aya"] else 41

    for layer_i in range(layer_num):
        if (intervention_type == 'type-1' and layer_i in [ _ for _ in range(21, layer_num)]) or (intervention_type == 'type-2' and layer_i in [ _ for _ in range(21)]):
            continue
        lang_sim_matrix = np.zeros((len(langs), len(langs)))

        for i, lang1 in enumerate(langs):
            for j, lang2 in enumerate(langs):
                if lang1 == lang2:
                    lang_sim_matrix[i, j] = 1.0
                else:
                    key = f"{lang1}-{lang2}"
                    key_rev = f"{lang2}-{lang1}"

                    if key in sim_dict and layer_i in sim_dict[key]:
                        sims = sim_dict[key][layer_i]
                        lang_sim_matrix[i, j] = np.mean(sims)
                    elif key_rev in sim_dict and layer_i in sim_dict[key_rev]:
                        sims = sim_dict[key_rev][layer_i]
                        lang_sim_matrix[i, j] = np.mean(sims)
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
            vmax=0.5,
            square=True,
            annot_kws={"size": 20}
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        title = 'LLaMA3-8B' if model_type == 'llama3' else 'Mistral-7B' if model_type == 'mistral' else 'Aya expanse-8B' if model_type == 'aya' else 'Phi4-14B'
        plt.title(f"{title} - Layer {layer_i}", fontsize=30)
        plt.tick_params(labelsize=30)

        if intervention_type == 'normal':
            save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance"
        elif intervention_type == 'type-1':
            save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/type-1"
        if intervention_type == 'type-2':
            save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/{model_type}/distance/type-2"
        os.makedirs(save_dir, exist_ok=True)
        if layer_i == 0:
            save_path = f'{save_dir}/emb_layer'
        else:
            save_path = f"{save_dir}/layer_{layer_i}"

        with PdfPages(save_path + '.pdf') as pdf:
            pdf.savefig(bbox_inches='tight', pad_inches=0.01)
            plt.close()