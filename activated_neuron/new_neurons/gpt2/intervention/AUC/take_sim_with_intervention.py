import os
import sys
# sys.path.append("/home/s2410121/proj_LA/activated_neuron")
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/intervention")
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/intervention/AUC")
import random
import dill as pickle
from collections import defaultdict

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from intervention_funcs import (
    take_similarities_with_edit_activation,
    plot_hist,
    save_as_pickle,
    unfreeze_pickle,
)

if __name__ == "__main__":

    # L1 = english
    L1 = "en"
    """ model configs """
    # GPT-2-small
    model_names = {
        # "base": "gpt2",
        "ja": "rinna/japanese-gpt2-small", # ja
        "nl": "GroNLP/gpt2-small-dutch", # du
        "it": "GroNLP/gpt2-small-italian", # ita
        "ko": "skt/kogpt2-base-v2", # ko
        # "de": "ml6team/gpt2-small-german-finetune-oscar", # ger
        # "fr": "dbddv01/gpt2-french-small", # fre
        # "es": "datificate/gpt2-small-spanish" # spa
    }
    """ parameters """
    # activation_type = "abs"
    activation_type = "product"
    norm_type = "no"
    n_list = [100, 1000, 3000, 4000, 5000, 7000, 10000, 15000, 20000, 30000] # patterns of intervention_num
    # n_list = [100]

    for L2, model_name in model_names.items():

        """ shared_neuronsのうち、AP上位nコ """
        pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/sorted_neurons_{L2}.pkl"
        sorted_neurons_AP = unfreeze_pickle(pkl_file_path)
        print(f"top 10 AP: \n {sorted_neurons_AP[:10]}")

        """ tatoeba translation corpus """
        dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
        num_sentences = 2000
        dataset = dataset.select(range(num_sentences))
        tatoeba_data = []
        for item in dataset:
            # check if there are empty sentences.
            if item['translation'][L1] != '' and item['translation'][L2] != '':
                tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
        # tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]
        tatoeba_data_len = len(tatoeba_data)
        """
        baseとして、対訳関係のない1文ずつのペアを作成
        (L1(en)はhttps://huggingface.co/datasets/agentlans/high-quality-english-sentences,
        L2はtatoebaの該当データを使用)
        """
        random_data = []
        # L1(en)
        en_base_ds = load_dataset("agentlans/high-quality-english-sentences")
        random_data_en = en_base_ds["train"][:num_sentences]
        en_base_ds_idx = 0
        for item in dataset:
            random_data.append((random_data_en["text"][en_base_ds_idx], item["translation"][L2]))
            en_base_ds_idx += 1

        """ model and device configs """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = model_names[L2]
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for n in n_list:
            """ どのくらい介入するか(n) """
            intervention_num = n
            sorted_neurons_AP_main = sorted_neurons_AP[:intervention_num]
            # half_num = intervention_num // 2
            # sorted_neurons_AP_main = sorted_neurons_AP[:half_num] + sorted_neurons_AP[-half_num:]
            sorted_neurons_AP_baseline = random.sample(sorted_neurons_AP[intervention_num+1:], len(sorted_neurons_AP[intervention_num+1:]))
            sorted_neurons_AP_baseline = sorted_neurons_AP_baseline[:intervention_num]

            """ deactivate shared_neurons(same semantics expert neurons) """
            similarities_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_main, tatoeba_data)
            similarities_non_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_main, random_data)
            final_results_same_semantics = defaultdict(float)
            final_results_non_same_semantics = defaultdict(float)
            for layer_idx in range(12): # 1２ layers
                final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
                final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
            plot_hist(final_results_same_semantics, final_results_non_same_semantics, L2, "AUC", activation_type, f"n_{intervention_num}")

            """ deactivate shared_neurons(same semantics(including non_same_semantics)) """
            similarities_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, tatoeba_data)
            similarities_non_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, random_data)
            final_results_same_semantics = defaultdict(float)
            final_results_non_same_semantics = defaultdict(float)
            for layer_idx in range(12): # 1２ layers
                final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
                final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
            plot_hist(final_results_same_semantics, final_results_non_same_semantics, L2, "AUC_baseline", activation_type, f"n_{intervention_num}")

            print(f"intervention_num: {n} <- completed.")

        # delete model and some cache
        del model
        torch.cuda.empty_cache()
