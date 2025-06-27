import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons")
import random
import pickle
import itertools
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
def plot_hist_llama3(dict1, dict2, L1: str, L2: str, score_type: str, intervention_num: str, is_en=False, is_baseline=False) -> None:
    # convert keys and values into list
    keys = np.array(list(dict1.keys()))
    values1 = list(dict1.values())
    values2 = list(dict2.values())

    offset = 0.1 # バーをずらす用

    # plot hist
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.figure(figsize=(8, 7))
    plt.bar(keys-offset, values1, alpha=1, label='same semantics')
    plt.bar(keys+offset, values2, alpha=1, label='different semantics')

    plt.xlabel('Layer index', fontsize=35)
    plt.ylabel('Cosine Sim', fontsize=35)
    plt.ylim(-0.5, 1)
    plt.title(f'en-{L1}, {L2}-deactivated', fontsize=35)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.legend(fontsize=25)
    plt.grid(True)
    if is_en:
        if not is_baseline:
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/{model_type}/final/{score_type}/shuffle/en/input_{L1}_deact_{L2}_n{intervention_num}"
        elif is_baseline:
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/{model_type}/final/{score_type}/shuffle/en/baseline/input_{L1}_deact_{L2}_n{intervention_num}"
    elif not is_en:
        if not is_baseline:
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/{model_type}/final/{score_type}/shuffle/input_{L1}_deact_{L2}_n{intervention_num}"
            # path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/{model_type}/final/{score_type}/shuffle/reverse/input_{L1}_deact_{L2}_n{intervention_num}"
        elif is_baseline:
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/{model_type}/final/{score_type}/shuffle/baseline/input_{L1}_deact_{L2}_n{intervention_num}"
            # path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/{model_type}/final/{score_type}/shuffle/reverse/baseline/input_{L1}_deact_{L2}_n{intervention_num}"
    with PdfPages(path + '.pdf') as pdf:
        pdf.savefig(bbox_inches='tight', pad_inches=0.01)
        plt.close()
        
if __name__ == "__main__":
    # L1 = "en"
    """ model configs """
    model_names = ['meta-llama/Meta-Llama-3-8B', 'CohereForAI/aya-expanse-8b', 'mistralai/Mistral-7B-v0.3', 'bigscience/bloom-3b']
    model_names = ['CohereForAI/aya-expanse-8b', 'mistralai/Mistral-7B-v0.3', 'meta-llama/Meta-Llama-3-8B']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    """ parameters """
    langs = ['ja', 'nl', 'ko', 'it', 'vi', 'ru', 'fr']
    # langs = ['nl', 'ja', 'ko', 'it']
    # n_list = [100, 1000, 3000, 5000]
    n_list = [100, 1000]
    score_types = ["cos_sim", "L2_dis"]
    score_types = ['cos_sim']
    is_en = False

    for model_name in model_names:
        model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya' if  'aya' in model_name else 'bloom'
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for L1, L2 in itertools.permutations(langs, 2): # L1: input language, L2: language to be deactivated.
            """ tatoeba translation corpus """
            dataset = load_dataset("tatoeba", lang1='en', lang2=L1, split="train")
            # # select first 2000 sentences.
            total_sentence_num = 2000 if L2 == "ko" else 5000
            num_sentences = 1000
            dataset = dataset.select(range(total_sentence_num))

            # same semantics sentence pairs: test split.
            tatoeba_data = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L1}_multi_test.pkl")

            """ non-translation pair for baseline. """
            random_data = []
            if L1 == "ko": # koreanはデータ数が足りない
                dataset2 = load_dataset("tatoeba", lang1='en', lang2='ja', split="train").select(range(5000))
            for sentence_idx, item in enumerate(dataset):
                if sentence_idx == num_sentences: break
                if L1 == "ko" and dataset2['translation'][num_sentences+sentence_idx]['en'] != '' and item['translation'][L1] != '':
                    random_data.append((dataset2["translation"][num_sentences+sentence_idx]['en'], item["translation"][L1])) 
                elif L1 != "ko" and dataset['translation'][num_sentences+sentence_idx]['en'] != '' and item['translation'][L1] != '':
                    random_data.append((dataset["translation"][num_sentences+sentence_idx]['en'], item["translation"][L1]))
            
            for score_type in score_types:
                save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
                # save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
                sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
                sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(0, 20)]]
                # sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
                
                for n in n_list:
                    """ n: intervention_num """
                    intervention_num = n
                    sorted_neurons_AP_main = sorted_neurons[:n]
                    random.seed(42)
                    sorted_neurons_AP_baseline = random.sample(sorted_neurons[intervention_num:], intervention_num)
                    """ deactivate shared_neurons(same semantics expert neurons) """
                    similarities_same_semantics = take_similarities_with_edit_activation(model, model_type, tokenizer, device, sorted_neurons_AP_main, tatoeba_data)
                    similarities_non_same_semantics = take_similarities_with_edit_activation(model, model_type, tokenizer, device, sorted_neurons_AP_main, random_data)
                    final_results_same_semantics = defaultdict(float)
                    final_results_non_same_semantics = defaultdict(float)
                    for layer_idx in range(model.config.num_hidden_layers): # ３２ layers
                        final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
                        final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
                    plot_hist_llama3(final_results_same_semantics, final_results_non_same_semantics, L1, L2, score_type, intervention_num, is_en)

                    # """ baseline """
                    # similarities_same_semantics = take_similarities_with_edit_activation(model, model_type, tokenizer, device, sorted_neurons_AP_baseline, tatoeba_data)
                    # similarities_non_same_semantics = take_similarities_with_edit_activation(model, model_type, tokenizer, device, sorted_neurons_AP_baseline, random_data)
                    # final_results_same_semantics = defaultdict(float)
                    # final_results_non_same_semantics = defaultdict(float)
                    # for layer_idx in range(model.config.num_hidden_layers):
                    #     final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
                    #     final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
                    # plot_hist_llama3(final_results_same_semantics, final_results_non_same_semantics, L1, L2, score_type, intervention_num, is_en, True)

                    print(f"input: {L1}, deactivated: {L2} intervention_num: {n} <- completed.")
    
        del model
        torch.cuda.empty_cache()