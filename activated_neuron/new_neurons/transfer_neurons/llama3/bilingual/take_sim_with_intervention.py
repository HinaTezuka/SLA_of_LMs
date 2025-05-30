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
    take_similarities_with_edit_activation,
    plot_hist,
    save_as_pickle,
    unfreeze_pickle,
    take_similarities_with_edit_activation,
)

# visualization
def plot_hist_llama3(dict1: defaultdict(float), dict2: defaultdict(float), L2: str, AUC_or_AUC_baseline:str, intervention_num: str) -> None:
    # convert keys and values into list
    keys = np.array(list(dict1.keys()))
    values1 = list(dict1.values())
    values2 = list(dict2.values())

    offset = 0.1 # バーをずらす用

    # plot hist
    plt.bar(keys-offset, values1, alpha=1, label='same semantics')
    plt.bar(keys+offset, values2, alpha=1, label='different semantics')
    # plt.bar(keys, values1, alpha=1, label='same semantics')
    # plt.bar(keys, values2, alpha=1, label='different semantics')

    plt.xlabel('Layer index', fontsize=35)
    plt.ylabel('Cosine Sim', fontsize=35)
    plt.title(f'en_{L2}')
    plt.tick_params(axis='x', labelsize=15)  # x軸の目盛りフォントサイズ
    plt.tick_params(axis='y', labelsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/llama3/bilingual/{L2}_n{intervention_num}.png",
        bbox_inches="tight"
    )
    plt.close()

if __name__ == "__main__":

    # L1 = english
    L1 = "en"
    """ model configs """
    # LLaMA-3
    model_name = "meta-llama/Meta-Llama-3-8B"
    """ model and device configs """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    """ parameters """
    langs = ["ja", "nl", "it", "ko"]
    langs = ["ja", "nl"]
    norm_type = "no"
    n_list = [100, 1000, 1500] # patterns of intervention_num
    n_list = [10000]

    for L2 in langs:
        """ tatoeba translation corpus """
        dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
        # select first 2000 sentences.
        total_sentence_num = 2000 if L2 == "ko" else 5000
        num_sentences = 20
        dataset = dataset.select(range(total_sentence_num))
        tatoeba_data = []
        for sentence_idx, item in enumerate(dataset):
            if sentence_idx == num_sentences: break
            # check if there are empty sentences.
            if item['translation'][L1] != '' and item['translation'][L2] != '':
                tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
        tatoeba_data_len = len(tatoeba_data)

        """
        non-translation pair for baseline.
        """
        random_data = []
        if L2 == "ko": # koreanはデータ数が足りない
            dataset2 = load_dataset("tatoeba", lang1=L1, lang2="ja", split="train").select(range(5000))
        for sentence_idx, item in enumerate(dataset):
            if sentence_idx == num_sentences: break
            if L2 == "ko" and dataset2['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
                random_data.append((dataset2["translation"][num_sentences+sentence_idx][L1], item["translation"][L2])) 
            elif L2 != "ko" and dataset['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
                random_data.append((dataset["translation"][num_sentences+sentence_idx][L1], item["translation"][L2]))

        save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/bilingual/ap_lang_specific/sorted_neurons_{L2}_last_token.pkl"
        sorted_neurons_AP = unfreeze_pickle(save_path_sorted_neurons)
        
        for n in n_list:
            """ n: intervention_num """
            intervention_num = n
            # sorted_neurons_AP_main = sorted_neurons_AP[:1000] + sorted_neurons_AP[-1000:]
            sorted_neurons_AP_main = sorted_neurons_AP[:intervention_num]
            # half_num = intervention_num // 2
            # sorted_neurons_AP_main = sorted_neurons_AP[:half_num] + sorted_neurons_AP[-half_num:]
            # sorted_neurons_AP_baseline = random.sample(sorted_neurons_AP[intervention_num+1:], len(sorted_neurons_AP[intervention_num+1:]))
            # sorted_neurons_AP_baseline = sorted_neurons_AP_baseline[:intervention_num]

            """ deactivate shared_neurons(same semantics expert neurons) """
            similarities_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_main, tatoeba_data)
            similarities_non_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_main, random_data)
            final_results_same_semantics = defaultdict(float)
            final_results_non_same_semantics = defaultdict(float)
            for layer_idx in range(32): # ３２ layers
                final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
                final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
            plot_hist_llama3(final_results_same_semantics, final_results_non_same_semantics, L2, "AUC", f"n_{intervention_num}")

            """ deactivate shared_neurons(same semantics(including non_same_semantics)) """
            # similarities_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, tatoeba_data)
            # similarities_non_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, random_data)
            # final_results_same_semantics = defaultdict(float)
            # final_results_non_same_semantics = defaultdict(float)
            # for layer_idx in range(32): # ３２ layers
            #     final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
            #     final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
            # plot_hist_llama3(final_results_same_semantics, final_results_non_same_semantics, L2, "AUC_baseline", activation_type, norm_type, f"n_{intervention_num}")

            print(f"intervention_num: {n} <- completed.")
