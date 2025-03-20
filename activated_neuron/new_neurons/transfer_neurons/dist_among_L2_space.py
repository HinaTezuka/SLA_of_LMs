import os
import sys
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import zscore
import itertools
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    unfreeze_pickle,
    save_as_pickle,
)

langs = ["ja", "nl", "ko", "it"]
models = {
    'llama3': 'meta-llama/Meta-Llama-3-8B',
    'mistral': 'mistralai/Mistral-7B-v0.3',
}

device = "cuda" if torch.cuda.is_available() else "cpu"
num_layers = 32

# centroids of english texts.
centroids = {} # { L2: [shared_centroids(en-L2)_1, ...} <- len(values) = 32(layer_num).

# def normalize_l2(vec):
#     norm = np.linalg.norm(vec)  # L2ノルムを計算
#     return vec / norm if norm != 0 else vec  # 0割り防止
def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2

for model_type, model_name in models.items():
    # unfreeze centroids as pkl.
    path_ja = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_ja.pkl"
    c_ja = unfreeze_pickle(path_ja)
    path_nl = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_nl.pkl"
    c_nl = unfreeze_pickle(path_nl)
    path_ko = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_ko.pkl"
    c_ko = unfreeze_pickle(path_ko)
    path_it = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_it.pkl"
    c_it = unfreeze_pickle(path_it)
    path_en = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_en.pkl'
    c_en = unfreeze_pickle(path_en)

    languages = ["ja", "nl", "ko", "it", "en"]
    c_vectors = {"ja": c_ja, "nl": c_nl, "ko": c_ko, "it": c_it, "en": c_en}

    # normalization.
    for lang, vectors in c_vectors.items():
        for layer_i in range(len(vectors)):
            vectors[layer_i] = normalize(vectors[layer_i])

    # Create language pairs (e.g., "ja-nl", "ja-ko", ...)
    language_pairs = list(itertools.combinations(languages, 2))

    # Initialize an array to store distance data for 32 layers
    distance_data = np.zeros((32, len(language_pairs)))

    # dists = {}
    # Compute distances for each layer
    for layer in range(32):
        layer_vectors = np.array([c_vectors[lang][layer] for lang in languages])  # Get vectors for each language
        dist_matrix = cdist(layer_vectors, layer_vectors, metric='euclidean')  # Compute L2 distance

        # Extract distances for language pairs
        for idx, (lang1, lang2) in enumerate(language_pairs):
            i, j = languages.index(lang1), languages.index(lang2)
            distance_data[layer, idx] = dist_matrix[i, j]
            # dists[f'{lang1}_{lang2}'] = dist_matrix[i, j]
    
    # pkl_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/lang_dists/distance_among_each_langs.pkl'
    # save_as_pickle(pkl_path, dists)

    df = pd.DataFrame(distance_data, columns=[f"{l1}-{l2}" for l1, l2 in language_pairs])

    """ zscore norm per layer. """
    # df_zscore = df.apply(zscore, axis=1)  # 各レイヤーごとに正規化

    # # Plot heatmap with Z-score normalization
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(df_zscore, cmap="coolwarm", annot=True, fmt=".2f", xticklabels=True, yticklabels=[f"{i+1}" for i in range(32)])
    # plt.xlabel("Language Pair")
    # plt.ylabel("Layer")
    # plt.title("Layer-wise Language Pair Distance Heatmap")
    """ """

    """ """
    # Plot heatmap
    # vmin = np.percentile(df.values, 5)  # 下位5%の値
    # vmax = np.percentile(df.values, 95)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, cmap="Reds", xticklabels=True, yticklabels=[f"{i+1}" for i in range(32)], annot=True)
    plt.xlabel("Language Pair")
    plt.ylabel("Layer")
    plt.title("Layer-wise Language Pair Distance Heatmap")
    save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/distance_among_langs/{model_type}_dist.png'
    # save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/distance_among_langs/{model_type}_dist_normed_per_layer.png'
    plt.savefig(
        save_path,
        bbox_inches='tight'
    )