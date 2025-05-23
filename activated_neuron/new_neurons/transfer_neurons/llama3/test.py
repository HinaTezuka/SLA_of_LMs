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
from baukit import TraceDict
from sklearn.metrics.pairwise import cosine_similarity

from funcs import (
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

    plt.xlabel('Layer index', fontsize=35)
    plt.ylabel('Cosine Sim', fontsize=35)
    plt.ylim(0, 1)
    plt.title(f'en_{L2}')
    plt.tick_params(axis='x', labelsize=15)  # x軸の目盛りフォントサイズ
    plt.tick_params(axis='y', labelsize=15)
    plt.legend()
    plt.grid(True)
    if is_en:
        if not is_baseline:
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/llama3/final/{score_type}/en/{L2}_n{intervention_num}.png"
        elif is_baseline:
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/llama3/final/{score_type}/en/baseline/{L2}_n{intervention_num}.png"
    elif not is_en:
        if not is_baseline:
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/llama3/final/{score_type}/{L2}_n{intervention_num}_ttt.png"
        elif is_baseline:
            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/llama3/final/{score_type}/baseline/{L2}_n{intervention_num}.png"
    plt.savefig(
        path,
        bbox_inches="tight"
    )
    plt.close()

def edit_activation(output, layer, layer_idx_and_neuron_idx):
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if str(layer_idx) in layer:  # layer名にlayer_idxが含まれているか確認
            output[:, :, neuron_idx] *= 0.5  # 指定されたニューロンの活性化値をゼロに設定

    return output

def take_similarities_with_edit_activation(model, tokenizer, device, layer_neuron_list, L2_txt):
    trace_layers = [f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:

        return calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, device, L2_txt)

def run_inference_with_intact_model(model, tokenizer, device, layer_neuron_list, data):
    similarities = defaultdict(list)
    for L1_txt, L2_txt in data:
        # L1(en)
        inputs_L1 = tokenizer(L1_txt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_L1 = model(**inputs_L1, output_hidden_states=True)
        all_hidden_states_L1 = output_L1.hidden_states
        last_token_index_L1 = inputs_L1["input_ids"].shape[1] - 1
        last_token_hidden_states_L1 = [
            layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy()
            for layer_hidden_state in all_hidden_states_L1
        ]
        # L2
        last_token_hidden_states_L2 = take_similarities_with_edit_activation(model, tokenizer, device, layer_neuron_list, L2_txt)

        # get cos_sim.
        similarities = calc_cosine_sim(last_token_hidden_states_L1, last_token_hidden_states_L2, similarities)
    
    return similarities

def calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, device, L2_txt):

    # hidden_states = defaultdict(torch.Tensor)
    inputs_L2 = tokenizer(L2_txt, return_tensors="pt").to(device)

    # get hidden_states
    with torch.no_grad():
        output_L2 = model(**inputs_L2, output_hidden_states=True)

    all_hidden_states_L2 = output_L2.hidden_states
    # 最後のtokenのhidden_statesのみ取得
    last_token_index_L2 = inputs_L2["input_ids"].shape[1] - 1

    last_token_hidden_states_L2 = [
        layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy()
        for layer_hidden_state in all_hidden_states_L2
    ]

    return last_token_hidden_states_L2

def calc_cosine_sim(last_token_hidden_states_L1: list, last_token_hidden_states_L2: list, similarities: defaultdict(float)) -> defaultdict(float):
    """
    calc similarity per layer.
    """
    for layer_idx, (hidden_state_L1, hidden_state_L2) in enumerate(zip(last_token_hidden_states_L1, last_token_hidden_states_L2)):
        sim = cosine_similarity(hidden_state_L1, hidden_state_L2)[0, 0] # <- [[0.50695133]] のようになっているので、数値の部分だけ抽出
        similarities[layer_idx].append(sim)

    return similarities

dataset1 = load_dataset("tatoeba", lang1='en', lang2='ja', split="train")
dataset2 = load_dataset("tatoeba", lang1='en', lang2='nl', split="train")
dataset3 = load_dataset("tatoeba", lang1='en', lang2='ko', split="train")
dataset4 = load_dataset("tatoeba", lang1='en', lang2='it', split="train")
for sentence_idx, item in enumerate(dataset1):
    print(item['translation']['en'], item['translation']['ja'])
    if sentence_idx == 10: break
for sentence_idx, item in enumerate(dataset2):
    print(item['translation']['en'], item['translation']['nl'])
    if sentence_idx == 10: break
for sentence_idx, item in enumerate(dataset3):
    print(item['translation']['en'], item['translation']['ko'])
    if sentence_idx == 10: break
for sentence_idx, item in enumerate(dataset4):
    print(item['translation']['en'], item['translation']['it'])
    if sentence_idx == 10: break
# for L2 in ['ja', 'nl', 'ko', 'it']:
#     tatoeba_data = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_multi_test.pkl")
#     print(tatoeba_data[:10])
sys.exit()

if __name__ == "__main__":
    L1 = "en"
    """ model configs """
    # LLaMA-3(8B)
    model_name = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    """ parameters """
    langs = ["ja", "nl", "it", "ko"]
    langs = ['ja']
    n_list = [100, 1000, 3000, 5000, 8000, 10000, 15000, 20000, 30000] # patterns of intervention_num
    # n_list = [100, 1000, 3000, 5000, 8000, 10000]
    n_list = [100, 1000]
    score_types = ["cos_sim", "L2_dis"]
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
        tatoeba_data = tatoeba_data[:20]
        random_data = random_data[:20]
        # print(len(tatoeba_data))
        # print(len(random_data))

        for score_type in score_types:
            save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/{score_type}/{L2}_mono_train.pkl"
            # sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
            save_path_sorted_neurons_nl = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/{score_type}/nl_mono_train.pkl"
            save_path_sorted_neurons_ja = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/{score_type}/ja_mono_train.pkl"
            sorted_neurons_nl = unfreeze_pickle(save_path_sorted_neurons_nl)[:10000]
            sorted_neurons_ja = unfreeze_pickle(save_path_sorted_neurons_ja)[:10000]
            sorted_neurons = remove_duplicates(sorted_neurons_nl, sorted_neurons_ja)
            # save_path_score_dict = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/{score_type}/{L2}_score_dict_mono_train.pkl"
            # score_dict = unfreeze_pickle(save_path_score_dict)
            
            for n in n_list:
                """ n: intervention_num """
                intervention_num = n
                sorted_neurons_AP_main = sorted_neurons[:n]
                # random.seed(42)
                # sorted_neurons_AP_baseline = random.sample(sorted_neurons_AP_main[intervention_num+1:], len(sorted_neurons_AP_main[intervention_num+1:]))
                # sorted_neurons_AP_baseline = sorted_neurons_AP_baseline[:intervention_num]

                """ deactivate shared_neurons(same semantics expert neurons) """
                similarities_same_semantics = run_inference_with_intact_model(model, tokenizer, device, sorted_neurons_AP_main, tatoeba_data)
                similarities_non_same_semantics = run_inference_with_intact_model(model, tokenizer, device, sorted_neurons_AP_main, random_data)
                final_results_same_semantics = defaultdict(float)
                final_results_non_same_semantics = defaultdict(float)
                for layer_idx in range(32): # ３２ layers
                    final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
                    final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
                plot_hist_llama3(final_results_same_semantics, final_results_non_same_semantics, L2, score_type, intervention_num, is_en)

                """ deactivate shared_neurons(same semantics(including non_same_semantics)) """
                # similarities_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, tatoeba_data)
                # similarities_non_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, random_data)
                # final_results_same_semantics = defaultdict(float)
                # final_results_non_same_semantics = defaultdict(float)
                # for layer_idx in range(32): # ３２ layers
                #     final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
                #     final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
                # plot_hist_llama3(final_results_same_semantics, final_results_non_same_semantics, L2, score_type, intervention_num, is_en, True)

                print(f"intervention_num: {n} <- completed.")