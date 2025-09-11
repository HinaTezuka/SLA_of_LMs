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

def plot_hist_llama3(dict1, dict2, L2: str, intervention_type, num) -> None:
    # convert keys and values into list
    keys = np.array(list(dict1.keys()))
    values1 = list(dict1.values())
    values2 = list(dict2.values())

    offset = 0.1 # バーをずらす用

    # plot hist
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.figure(figsize=(8, 7))
    plt.bar(keys-offset, values1, alpha=1, label='same semantics')
    plt.bar(keys+offset, values2, alpha=1, label='different semantics')

    plt.xlabel('Layer index', fontsize=35)
    plt.ylabel('Cosine Sim', fontsize=35)
    plt.ylim(-0.5, 1)
    plt.title(f'en-{L2}', fontsize=35)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.legend(fontsize=25)
    plt.grid(True)
    path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/subspace/test_figs/{model_type}_{L2}_{intervention_type}_{num}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with PdfPages(path + '.pdf') as pdf:
        pdf.savefig(bbox_inches='tight', pad_inches=0.01)
        plt.close()

langs = ["ja", "nl", "ko", "it", "en", "vi", "ru", "fr"]
# LLaMA3-8B / Mistral-7B / Aya-expanse-8B / Phi4-14B.
model_names = ['CohereForAI/aya-expanse-8b', "meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3"]
is_using_centroids = False
intervention_type = 'type-1'
model_type = None
num = 10000

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
        # hs_vi = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/vi.pkl")
        # hs_ru = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ru.pkl")
        # hs_fr = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/fr.pkl")
    elif intervention_type == 'type-1':
        hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ja_type1_{num}.pkl")
        hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/nl_type1_{num}.pkl")
        hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ko_type1_{num}.pkl")
        hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/it_type1_{num}.pkl")
        # hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ja_type1_TEST.pkl")
        # hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/nl_type1_TEST.pkl")
        # hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ko_type1_TEST.pkl")
        # hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/it_type1_TEST.pkl")
        hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/en_type1.pkl")
        # hs_vi = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/vi.pkl")
        # hs_ru = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ru.pkl")
        # hs_fr = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/fr.pkl")
    elif intervention_type == 'type-2':
        hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/ja.pkl")
        hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/nl.pkl")
        hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/ko.pkl")
        hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/it.pkl")
        hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/en.pkl")
        # hs_vi = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/vi.pkl")
        # hs_ru = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ru.pkl")
        # hs_fr = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/fr.pkl")

    sim_dict = defaultdict(lambda: defaultdict(list)) # {lang1-lang2: layer_idx: [cos_sim1, cos_sim2, ... , cos_sim1000} 1000: total_num_of_sample_sentences.

    # compute cosine_sim beween vectors in each subspace.
    lang2hs_layer = {
        "ja": hs_ja,
        "nl": hs_nl,
        "ko": hs_ko,
        "it": hs_it,
        "en": hs_en,
    }
    target_lang_pairs = [("ja", "en"), ("nl", "en"), ("ko", "en"), ("it", "en")]

    for lang1, lang2 in target_lang_pairs:
        pair_means = {}
        neg_means = {}

        for layer_i in range(layer_num):
            mat1 = lang2hs_layer[lang1][layer_i]  # shape: (n, d)
            mat2 = lang2hs_layer[lang2][layer_i]  # shape: (n, d)

            sim_matrix = cosine_similarity(mat1, mat2)
            n = sim_matrix.shape[0]

            # 正解ペア (i, i)
            pair_sims = np.diag(sim_matrix)
            # ランダムペア (i, j≠i)
            neg_sims = sim_matrix[~np.eye(n, dtype=bool)]

            pair_means[layer_i] = float(np.mean(pair_sims))
            neg_means[layer_i] = float(np.mean(neg_sims))

        plot_hist_llama3(
            dict1=pair_means,
            dict2=neg_means,
            L2=lang1,
            intervention_type=intervention_type,
            num=num,
        )

    # for layer_i in range(layer_num):
    #     hs_ja_layer = np.array(hs_ja[layer_i]) # shape: (n * d) n: sample_num, d: dimention of hs.
    #     hs_nl = np.array(hs_nl[layer_i])
    #     hs_ko_layer = np.array(hs_ko[layer_i])
    #     hs_it_layer = np.array(hs_it[layer_i])
    #     hs_en_layer = np.array(hs_en[layer_i])
    #     hs_vi_layer = np.array(hs_vi[layer_i])
    #     hs_ru_layer = np.array(hs_ru[layer_i])
    #     hs_fr_layer = np.array(hs_fr[layer_i])
        
    #     # compute cosine_sim beween vectors in each subspace.
    #     lang2hs_layer = {
    #         "ja": hs_ja_layer,
    #         "nl": hs_nl,
    #         "ko": hs_ko_layer,
    #         "it": hs_it_layer,
    #         "en": hs_en_layer,
    #         "vi": hs_vi_layer,
    #         "ru": hs_ru_layer,
    #         "fr": hs_fr_layer,
    #     }

    #     target_lang_pairs = [("ja", "en"), ("nl", "en"), ("ko", "en"), ("it", "en")]
    #     for lang1, lang2 in target_lang_pairs:

    #         mat1 = lang2hs_layer[lang1]  # shape: (1000, 4096)
    #         mat2 = lang2hs_layer[lang2]  # shape: (1000, 4096)

    #         sim_matrix = cosine_similarity(mat1, mat2)
    #         pair_sims = np.diag(sim_matrix)              # 正解ペア
    #         neg_sims = sim_matrix[~np.eye(n, dtype=bool)]  # 非ペア

    #         # 各レイヤーの平均を辞書に追加
    #         pair_means[layer_i] = float(np.mean(pair_sims))
    #         neg_means[layer_i] = float(np.mean(neg_sims))

    # plot_hist_llama3(
    #     dict1=pair_means,
    #     dict2=neg_means,
    #     L2=lang1,                  # 言語名
    #     score_type="cos_sim",     # スコアの種類
    #     intervention_num="1000"      # 介入番号など
    # )

            # sim_matrix = cosine_similarity(mat1, mat2)

            # # average similarity for each row in mat1 against all rows in mat2
            # avg_sims = np.mean(sim_matrix, axis=1)  # shape: (1000,)

            # """"""
            # n = sim_matrix.shape[0]
            # neg_sims = sim_matrix[~np.eye(n, dtype=bool)]  # shape: (n*(n-1),)

            # # # その中からランダムにサンプリング
            # # if layer_i == 20:
            # #     rand_neg_sims = np.random.choice(neg_sims, size=1000, replace=False)
            # #     print(np.mean(rand_neg_sims))
            # #     sys.exit()

            # pair_sims = np.diag(sim_matrix) 
            # if layer_i == 20:
            #     print(pair_sims.shape)  # (1000,)
            #     print(np.mean(pair_sims))
            #     # sys.exit()
