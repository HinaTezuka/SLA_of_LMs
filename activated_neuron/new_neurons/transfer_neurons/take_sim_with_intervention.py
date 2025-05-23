import os
import sys
import random
import pickle
from collections import defaultdict

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from funcs import (
    take_similarities_with_edit_activation,
    plot_hist,
    save_as_pickle,
    unfreeze_pickle,
    take_similarities_with_edit_activation,
)

if __name__ == "__main__":

    # L1 = english
    L1 = "en"
    """ model configs """
    # LLaMA-3
    model_names = {
        # "base": "meta-llama/Meta-Llama-3-8B"
        # "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
        # # "de": "DiscoResearch/Llama3-German-8B", # ger
        # "nl": "ReBatch/Llama-3-8B-dutch", # du
        # "it": "DeepMount00/Llama-3-8b-Ita", # ita
        "ko": "beomi/Llama-3-KoEn-8B", # ko
    }
    """ parameters """
    norm_type = "no"
    n_list = [100, 1000, 1500] # patterns of intervention_num
    n_list = [5000]

    for L2, model_name in model_names.items():

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

        """ model and device configs """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/sorted_neurons_{L2}_last_token.pkl"
        sorted_neurons_AP = unfreeze_pickle(save_path_sorted_neurons)
        
        for n in n_list:
            """ n: intervention_num """
            intervention_num = n
            sorted_neurons_AP_main = sorted_neurons_AP[:n] + sorted_neurons_AP[-n:]
            # sorted_neurons_AP_main = sorted_neurons_AP[:intervention_num]
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
            plot_hist(final_results_same_semantics, final_results_non_same_semantics, L2, "AUC", f"n_{intervention_num}")

            """ deactivate shared_neurons(same semantics(including non_same_semantics)) """
            # similarities_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, tatoeba_data)
            # similarities_non_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, random_data)
            # final_results_same_semantics = defaultdict(float)
            # final_results_non_same_semantics = defaultdict(float)
            # for layer_idx in range(32): # ３２ layers
            #     final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
            #     final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
            # plot_hist(final_results_same_semantics, final_results_non_same_semantics, L2, "AUC_baseline", activation_type, norm_type, f"n_{intervention_num}")

            print(f"intervention_num: {n} <- completed.")

        # delete model and some cache
        del model
        torch.cuda.empty_cache()
