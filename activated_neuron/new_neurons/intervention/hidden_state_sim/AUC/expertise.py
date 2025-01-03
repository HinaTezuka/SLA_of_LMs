"""
compute and extract "expert shared neurons" for: same semantics(translation pairs).
"""
import os
import itertools
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/hidden_state_sim/AUC")
import dill as pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import average_precision_score

from expertise_funcs import (
    track_neurons_with_text_data,
    save_as_pickle,
    unfreeze_pickle,
    compute_ap_and_sort,
)

""" parameters setting """
activation_type = "abs"
# activation_type = "product"
# norm_type = "no"
# norm_type = "min_max"
norm_type = "sigmoid"
# L2 = "ja"
L2_list = ["ja", "nl", "ko", "it"]

for L2 in L2_list:

    """ unfreeze activation_dicts """
    pkl_path_same_semantics = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/same_semantics/en_{L2}.pkl"
    pkl_path_non_same_semantics = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/non_same_semantics/en_{L2}.pkl"
    act_same_semantics_dict = unfreeze_pickle(pkl_path_same_semantics)
    act_non_same_semantics_dict = unfreeze_pickle(pkl_path_non_same_semantics)
    print(f"unfreezed pickles for {L2}.")

    """ calc AP and sort. """
    sorted_neurons, ap_scores = compute_ap_and_sort(act_same_semantics_dict, act_non_same_semantics_dict, norm_type)

    """ pickle operations and test outputs. """
    # save as pickle file
    sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/sorted_neurons_{L2}.pkl"
    ap_scores_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/ap_scores_{L2}.pkl"
    save_as_pickle(sorted_neurons_path, sorted_neurons)
    save_as_pickle(ap_scores_path, ap_scores)

    # unfreeze pickle
    sorted_neurons = unfreeze_pickle(sorted_neurons_path)
    ap_scores_path = unfreeze_pickle(ap_scores_path)

    print(f"{L2} <- completed.")
