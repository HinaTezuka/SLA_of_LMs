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
from sklearn.decomposition import PCA

from funcs import (
    unfreeze_pickle,
)

langs = ["ja", "nl", "ko", "it", "en"]
langs = ['nl']
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
            pca = PCA(random_state=42)
            pca.fit(hs_layer)
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_explained_variance = np.cumsum(explained_variance_ratio)
            if layer_i == 32:
                print(cumulative_explained_variance)
                sys.exit()