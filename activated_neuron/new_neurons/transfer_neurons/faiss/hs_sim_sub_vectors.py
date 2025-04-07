import os
import sys
import pickle

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

def calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, data, L2: str, is_sub:bool=False):
    """
    各層について、2000文ペアそれぞれのhidden_statesの類似度の平均を計算
    """
    similarities = defaultdict(list) # {layer_idx: mean_sim_of_each_sentences}
    """ load centroids. """
    c_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/centroids/c_train_en.pkl")
    c_L2 = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/centroids/c_train_{L2}.pkl")
    candidate_layers = [5]

    if is_sub:
        # hook_fn.
        def add_subtracted_vector(model, input, output, layer_idx: int):
            if layer_idx != 31:
                output[0][:, -1, :] += torch.from_numpy(c_en[layer_idx] - c_L2[layer_idx]).to(device)
        # register hook.
        handles = []
        for layer_idx, layer in enumerate(model.model.layers):
            # for adding subtracted vector to the hidden_states.
            if layer_idx in candidate_layers:
                handle = layer.register_forward_hook(
                    lambda model, input, output, layer_idx=layer_idx: add_subtracted_vector(model, input, output, layer_idx)
                )
                handles.append(handle)

    for L1_txt, L2_txt in data:
        hidden_states = defaultdict(torch.Tensor)
        inputs_L1 = tokenizer(L1_txt, return_tensors="pt").to(device)
        inputs_L2 = tokenizer(L2_txt, return_tensors="pt").to(device)

        # get hidden_states
        with torch.no_grad():
            output_L1 = model(**inputs_L1, output_hidden_states=True)
            output_L2 = model(**inputs_L2, output_hidden_states=True)

        all_hidden_states_L1 = output_L1.hidden_states[1:]
        all_hidden_states_L2 = output_L2.hidden_states[1:]

        last_token_index_L1 = inputs_L1["input_ids"].shape[1] - 1
        last_token_index_L2 = inputs_L2["input_ids"].shape[1] - 1

        last_token_hidden_states_L1 = [
            layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy() for layer_hidden_state in all_hidden_states_L1
        ]
        last_token_hidden_states_L2 = [
            layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy() for layer_hidden_state in all_hidden_states_L2
        ]
        
        # cos_sim
        similarities = calc_cosine_sim(last_token_hidden_states_L1, last_token_hidden_states_L2, similarities)

    if is_sub:
        # remove hook
        for handle in handles:
            handle.remove()

    return similarities

def calc_cosine_sim(last_token_hidden_states_L1: list, last_token_hidden_states_L2: list, similarities: defaultdict(float)) -> defaultdict(float):
    """
    層ごとの類似度を計算
    """
    for layer_idx, (hidden_state_L1, hidden_state_L2) in enumerate(zip(last_token_hidden_states_L1, last_token_hidden_states_L2)):
        sim = cosine_similarity(hidden_state_L1, hidden_state_L2)[0, 0] # <- [[0.50695133]] のようになっているので、数値の部分だけ抽出
        similarities[layer_idx].append(sim)

    return similarities

def plot_hist(dict1: defaultdict(float), dict2: defaultdict(float), L2: str) -> None:
    # convert keys and values into list
    keys = np.array(list(dict1.keys()))
    # keys = list(dict1.keys())
    values1 = list(dict1.values())
    values2 = list(dict2.values())

    offset = 0.1

    # plot hist
    plt.bar(keys-offset, values1, alpha=1, label='same semantics')
    plt.bar(keys+offset, values2, alpha=1, label='different semantics')

    plt.xlabel('Layer index', fontsize=35)
    plt.ylabel('Cosine Sim', fontsize=35)
    plt.title(f'en_{L2}')
    plt.tick_params(axis='x', labelsize=15)  # x軸の目盛りフォントサイズ
    plt.tick_params(axis='y', labelsize=15)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/llama3/sub/{L2}.png",
        bbox_inches="tight",
        )
    plt.close()

def unfreeze_pickle(file_path: str):
    """
    Load a pickle file as a dictionary with error handling.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Error unpickling file {file_path}: {e}")

if __name__ == "__main__":
    """ model configs """
    langs = ["ja", "nl", "ko", "it"]
    # original llama
    model_name = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    for L2 in langs:
        L1 = "en" # L1 is fixed to english.
        """ tatoeba translation corpus """
        dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
        # select first 2000 sentences
        total_sentence_num = 2000 if L2 == "ko" else 5000
        num_sentences = 2000
        dataset = dataset.select(range(total_sentence_num))
        tatoeba_data = []
        for sentence_idx, item in enumerate(dataset):
            if sentence_idx == num_sentences: break
            # check if there are empty sentences.
            if item['translation'][L1] != '' and item['translation'][L2] != '':
                tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
        tatoeba_data_len = len(tatoeba_data)

        """
        baseとして、対訳関係のない1文ずつのペアを作成.
        """
        random_data = []
        if L2 == "ko":
            dataset2 = load_dataset("tatoeba", lang1=L1, lang2="ja", split="train").select(range(5000))
        for sentence_idx, item in enumerate(dataset):
            if sentence_idx == num_sentences: break
            if L2 == "ko" and dataset2['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
                random_data.append((dataset2["translation"][num_sentences+sentence_idx][L1], item["translation"][L2])) 
            elif L2 != "ko" and dataset['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
                random_data.append((dataset["translation"][num_sentences+sentence_idx][L1], item["translation"][L2]))
        tatoeba_data = tatoeba_data[:10]
        random_data = random_data[:10]

        """ calc similarities """
        results_same_semantics = calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, tatoeba_data, L2, is_sub=True)
        results_non_same_semantics = calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, random_data, L2)
        final_results_same_semantics = defaultdict(float)
        final_results_non_same_semantics = defaultdict(float)
        for layer_idx in range(32):
            final_results_same_semantics[layer_idx] = np.array(results_same_semantics[layer_idx]).mean()
            final_results_non_same_semantics[layer_idx] = np.array(results_non_same_semantics[layer_idx]).mean()

        torch.cuda.empty_cache()

        """ plot """
        plot_hist(final_results_same_semantics, final_results_non_same_semantics, L2)

    print("visualization completed !")    