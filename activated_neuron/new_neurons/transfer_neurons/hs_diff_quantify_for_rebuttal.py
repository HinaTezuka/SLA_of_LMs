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
)

# langs = ["ja", "nl", "ko", "it", "en", "vi", "ru", "fr"]
langs = ["ja", "nl", "ko", "it", "en"]
# LLaMA3-8B / Mistral-7B / Aya-expanse-8B / BLOOM-3B.
model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b']
model_type = 'llama3'
L2 = 'nl'

is_en_only = False

if not is_en_only:
    path_same = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/cos_sim/hs_sim/en_{L2}_type1_same_semantics.pkl'
    # path_same = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/cos_sim/hs_sim/normal_same_semantics_{L2}.pkl'
    same = unfreeze_pickle(path_same)
    path_diff = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/cos_sim/hs_sim/en_{L2}_type1_non_same_semantics.pkl'
    # path_diff = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/cos_sim/hs_sim/normal_non_same_semantics_{L2}.pkl'
    diff = unfreeze_pickle(path_diff)
elif is_en_only:
    path_same = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/cos_sim/hs_sim/en_only_type1_same_semantics_{L2}.pkl'
    same = unfreeze_pickle(path_same)
    path_diff = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/cos_sim/hs_sim/en_only_type1_non_same_semantics_{L2}.pkl'
    diff = unfreeze_pickle(path_diff)

for layer_i in range(32):
    if layer_i == 0: print(f'==================== {"Normal"} ====================')
    """  """
    print(f'--------------- {layer_i} ----------------')
    print(abs(same[layer_i] - diff[layer_i]))
