import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from collections import defaultdict

import numpy as np

from funcs import (
    multilingual_dataset_for_lang_specific_detection,
    track_neurons_with_text_data,
    save_as_pickle,
    compute_ap_and_sort,
    unfreeze_pickle,
    save_np_arrays,
    unfreeze_np_arrays,
)

langs = ['ja', 'nl', 'ko', 'it']
model = 'mistral'
model = 'llama3'
score_types = ['cos_sim', 'L2_dis']

def intersection_ratio(*lists):
    # Convert each list to a set
    sets = [set(lst) for lst in lists]
    
    # Compute the intersection (common elements)
    intersection = set.intersection(*sets)
    
    # Compute the union (all unique elements)
    union = set.union(*sets)
    
    # Calculate Jaccard index
    jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0
    
    # Calculate the ratio of intersection size to the average list size
    avg_list_size = sum(len(s) for s in sets) / len(sets)
    intersection_ratio = len(intersection) / avg_list_size if avg_list_size > 0 else 0
    
    return {
        "Number of common elements": len(intersection),
        "Jaccard index": jaccard_index,
        "Intersection ratio": intersection_ratio,
        # "Common elements": intersection
    }

top_n = 1000
for score_type in score_types:
    # top score neurons
    # ja
    # each language-specific space -> shared semantic space.
    # ja_save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/{score_type}/ja_revised.pkl"
    # ja_sorted_neurons = unfreeze_pickle(ja_save_path_sorted_neurons)[:top_n]
    # nl_save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/{score_type}/nl_revised.pkl"
    # nl_sorted_neurons = unfreeze_pickle(nl_save_path_sorted_neurons)[:top_n]
    # ko_save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/{score_type}/ko_revised.pkl"
    # ko_sorted_neurons = unfreeze_pickle(ko_save_path_sorted_neurons)[:top_n]
    # it_save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/{score_type}/it_revised.pkl"
    # it_sorted_neurons = unfreeze_pickle(it_save_path_sorted_neurons)[:top_n]

    # reverse.
    ja_save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/reverse/{score_type}/ja_sorted_neurons.pkl"
    ja_sorted_neurons = unfreeze_pickle(ja_save_path_sorted_neurons)
    ja_sorted_neurons = [neuron for neuron in ja_sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
    ja_sorted_neurons = ja_sorted_neurons[:top_n]
    nl_save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/reverse/{score_type}/nl_sorted_neurons.pkl"
    nl_sorted_neurons = unfreeze_pickle(nl_save_path_sorted_neurons)
    nl_sorted_neurons = [neuron for neuron in nl_sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
    nl_sorted_neurons = nl_sorted_neurons[:top_n]
    ko_save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/reverse/{score_type}/ko_sorted_neurons.pkl"
    ko_sorted_neurons = unfreeze_pickle(ko_save_path_sorted_neurons)
    ko_sorted_neurons = [neuron for neuron in ko_sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
    ko_sorted_neurons = ko_sorted_neurons[:top_n]
    it_save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/reverse/{score_type}/it_sorted_neurons.pkl"
    it_sorted_neurons = unfreeze_pickle(it_save_path_sorted_neurons)
    it_sorted_neurons = [neuron for neuron in it_sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
    it_sorted_neurons = it_sorted_neurons[:top_n]

    print(f'{score_type}')
    print(f'ALL: {intersection_ratio(ja_sorted_neurons, nl_sorted_neurons, ko_sorted_neurons, it_sorted_neurons)}')
    print(f'ja-nl: {intersection_ratio(ja_sorted_neurons, nl_sorted_neurons)}')
    print(f'ja-ko: {intersection_ratio(ja_sorted_neurons, ko_sorted_neurons)}')
    print(f'nl-it: {intersection_ratio(nl_sorted_neurons, it_sorted_neurons)}')
    print(f'ja-it: {intersection_ratio(ja_sorted_neurons, it_sorted_neurons)}')