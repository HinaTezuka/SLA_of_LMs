import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons")
import random
import pickle
from collections import defaultdict

import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from funcs import (
    unfreeze_pickle,
)

def get_norm_of_value_vector(model, neuron: tuple) -> float:
    # if model == llama3 or mistral
    layer_idx, neuron_idx = neuron[0], neuron[1]
    value_vector = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[neuron_idx].detach().cpu().numpy()
    
    return np.linalg.norm(value_vector)

if __name__ == "__main__":
    L1 = "en"
    """ model configs """
    # LLaMA-3(8B)
    model_name = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    """ parameters """
    langs = ["ja", "nl", "it", "ko"]
    n_list = [100, 1000, 3000, 5000, 8000, 10000, 15000, 20000, 30000] # patterns of intervention_num
    score_types = ["cos_sim", "L2_dis"]
    top_n = 1000

    for L2 in langs:
        for score_type in score_types:
            save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/{score_type}/{L2}_revised.pkl"
            sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)[:top_n]
            # save_path_score_dict = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/{score_type}/{L2}_score_dict_revised.pkl"
            # score_dict = unfreeze_pickle(save_path_score_dict)

            norm_list = []
            for neuron in sorted_neurons:
                norm_list.append(get_norm_of_value_vector(model, neuron))
            
            print(f'{L2}, {score_type}')
            print(np.mean(np.array(norm_list)))
            # sys.exit()
