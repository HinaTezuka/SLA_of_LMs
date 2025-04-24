import os
import sys
import random
import pickle

import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from funcs import (
    take_similarities_with_edit_activation,
    plot_hist,
    save_as_pickle,
    unfreeze_pickle,
)

is_reverse = True
model_types = ['llama3', 'mistral', 'aya']
score_type = 'cos_sim'
langs = ['ja', 'nl', 'ko', 'it']

for model_type in model_types:
    for L2 in langs:
        path_transfers = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
        if is_reverse:
            path_transfers = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
        transfer_neurons = unfreeze_pickle(path_transfers)

        if is_reverse:
            transfer_neurons = [neuron for neuron in transfer_neurons if neuron[0] in [ _ for _ in range(20, 32)]][:1000]
        else:
            transfer_neurons = [neuron for neuron in transfer_neurons if neuron[0] in [ _ for _ in range(20)]][:1000]

        # lang_specific neurons
        path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/ap_lang_specific/sorted_neurons_{L2}.pkl"
        lang_specific_neurons = unfreeze_pickle(path)
        lang_specific_neurons = lang_specific_neurons[:1000]+lang_specific_neurons[-1000:]

        # def intersection_ratio(*lists):
        #     # Convert each list to a set
        #     sets = [set(lst) for lst in lists]
            
        #     # Compute the intersection (common elements)
        #     intersection = set.intersection(*sets)
            
        #     # Compute the union (all unique elements)
        #     union = set.union(*sets)
            
        #     # Calculate Jaccard index
        #     jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0
            
        #     # Calculate the ratio of intersection size to the average list size
        #     avg_list_size = sum(len(s) for s in sets) / len(sets)
        #     intersection_ratio = len(intersection) / avg_list_size if avg_list_size > 0 else 0
            
        #     return {
        #         "Number of common elements": len(intersection),
        #         "Jaccard index": jaccard_index,
        #         "Intersection ratio": intersection_ratio,
        #         # "Common elements": intersection
        #     }
        set_A = set(transfer_neurons)
        set_B = set(lang_specific_neurons)
        intersection = set_A & set_B
        num_overlap = len(intersection)
        overlap_rate = num_overlap / len(set_A)

        print(f'================================ {model_type, L2} ================================')
        print(overlap_rate)