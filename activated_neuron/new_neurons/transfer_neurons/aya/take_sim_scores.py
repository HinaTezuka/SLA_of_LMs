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
from matplotlib.backends.backend_pdf import PdfPages
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from funcs import (
    take_similarities_with_edit_activation,
    plot_hist,
    save_as_pickle,
    unfreeze_pickle,
)

# visualization
def plot_hist_llama3(dict1: defaultdict(float), dict2: defaultdict(float), L2: str, score_type: str, intervention_num: str, is_en=False, is_baseline=False) -> None:
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

    plt.xlabel('Layer index', fontsize=30)
    plt.ylabel('Cosine Sim', fontsize=30)
    plt.ylim(0, 1)
    plt.title(f'en_{L2}')
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.legend()
    plt.grid(True)
    if is_en:
        if not is_baseline:
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/aya/final/{score_type}/en/{L2}_n{intervention_num}"
        elif is_baseline:
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/aya/final/{score_type}/en/baseline/{L2}_n{intervention_num}"
    elif not is_en:
        if not is_baseline:
            # path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/aya/final/{score_type}/{L2}_n{intervention_num}"
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/aya/final/{score_type}/reverse/{L2}_n{intervention_num}"
        elif is_baseline:
            # path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/aya/final/{score_type}/baseline/{L2}_n{intervention_num}"
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/aya/final/{score_type}/reverse/baseline/{L2}_n{intervention_num}"
    # plt.savefig(
    #     path,
    #     bbox_inches="tight"
    # )
    with PdfPages(path + '.pdf') as pdf:
        pdf.savefig(bbox_inches='tight', pad_inches=0.01)
        plt.close()

if __name__ == "__main__":
    L1 = "en"
    """ model configs """
    # aya-expanse-8B
    model_name = 'CohereForAI/aya-expanse-8b'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    """ parameters """
    langs = ["ja", "nl", "it", "ko"]
    n_list = [100, 1000, 3000, 5000]
    # score_types = ["cos_sim", "L2_dis"]
    score_types = ['cos_sim']
    is_en = False

    for L2 in langs:
        """ tatoeba translation corpus """
        dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
        # # select first 2000 sentences.
        total_sentence_num = 2000 if L2 == "ko" else 5000
        num_sentences = 1000
        dataset = dataset.select(range(total_sentence_num))
        # tatoeba_data = []
        # for sentence_idx, item in enumerate(dataset):
        #     if sentence_idx == num_sentences: break
        #     # check if there are empty sentences.
        #     if item['translation'][L1] != '' and item['translation'][L2] != '':
        #         tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
        # tatoeba_data_len = len(tatoeba_data)

        # same semantics sentence pairs: test split.
        tatoeba_data = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_multi_test.pkl")

        """ non-translation pair for baseline. """
        random_data = []
        if L2 == "ko": # koreanはデータ数が足りない
            dataset2 = load_dataset("tatoeba", lang1=L1, lang2="ja", split="train").select(range(5000))
        for sentence_idx, item in enumerate(dataset):
            if sentence_idx == num_sentences: break
            if L2 == "ko" and dataset2['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
                random_data.append((dataset2["translation"][num_sentences+sentence_idx][L1], item["translation"][L2])) 
            elif L2 != "ko" and dataset['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
                random_data.append((dataset["translation"][num_sentences+sentence_idx][L1], item["translation"][L2]))

        def remove_duplicates(lista, listb):
            return [item for item in lista if item not in set(listb)]

        for score_type in score_types:
            # save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/aya/final_scores/{score_type}/{L2}_mono_train.pkl"
            save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/aya/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
            sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
            # sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(0, 20)]]
            sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
            
            for n in n_list:
                """ n: intervention_num """
                intervention_num = n
                sorted_neurons_AP_main = sorted_neurons[:n]
                random.seed(42)
                sorted_neurons_AP_baseline = random.sample(sorted_neurons[intervention_num:], intervention_num)

                """ deactivate shared_neurons(same semantics expert neurons) """
                similarities_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_main, tatoeba_data)
                similarities_non_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_main, random_data)
                final_results_same_semantics = defaultdict(float)
                final_results_non_same_semantics = defaultdict(float)
                for layer_idx in range(32): # ３２ layers
                    final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
                    final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
                plot_hist_llama3(final_results_same_semantics, final_results_non_same_semantics, L2, score_type, intervention_num, is_en)

                """ baseline """
                similarities_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, tatoeba_data)
                similarities_non_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, random_data)
                final_results_same_semantics = defaultdict(float)
                final_results_non_same_semantics = defaultdict(float)
                for layer_idx in range(32): # ３２ layers
                    final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
                    final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
                plot_hist_llama3(final_results_same_semantics, final_results_non_same_semantics, L2, score_type, intervention_num, is_en, True)

                print(f"{L2}, intervention_num: {n} <- completed.")