import os
import sys
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
    monolingual_dataset,
    get_norms_of_value_vector,
    unfreeze_pickle,
    save_as_pickle,
)

if __name__ == "__main__":
    L1 = "en"
    """ model configs """
    # LLaMA-3(8B)
    model_names = {
        "llama3": "meta-llama/Meta-Llama-3-8B",
        "mistral": "mistralai/Mistral-7B-v0.3",
    }
    for model_type in model_names.keys():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_names[model_type]).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_names[model_type])
        """ parameters """
        langs = ["ja", "nl", "it", "ko"]
        score_types = ["cos_sim", "L2_dis"]
        top_n = 100
        num_sentences = 1000

        results = defaultdict()
        for L2 in langs:
            # get monolingual dataset.
            data_mono = monolingual_dataset(L2, num_sentences)
            for score_type in score_types:
                save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_revised.pkl"
                sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
                sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(1)]]
                sorted_neurons = sorted_neurons[:top_n]
                # sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)[:top_n]
                # save_path_score_dict = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/{score_type}/{L2}_score_dict_revised.pkl"
                # score_dict = unfreeze_pickle(save_path_score_dict)
                
                res = get_norms_of_value_vector(model, tokenizer, device, sorted_neurons, data_mono)
                results[(L2, score_type)] = res

        # save as pkl.
        save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/norms/norms_{top_n}.pkl"
        # save_as_pickle(save_path, results)
        # results = unfreeze_pickle(save_path)

        # print results.
        print(f"\n ============================= {model_type}, nums_of_text: {num_sentences}, top_n: {top_n} ============================= \n")
        for L2 in langs:
            for score_type in score_types:
                res = results[(L2, score_type)]
                for k, v in res.items():
                    l = []
                    l.append(v)
                score = np.mean(np.array(l))
                print(f"{L2, score_type}: {score}")

        del model
        torch.cuda.empty_cache()