"""
モデルの各層のhidden stateを取得・対訳ペアと非対訳ペアでそれぞれ類似度を測定
"""
import os
import sys
# sys.path.append("/home/s2410121/proj_LA/activated_neuron")
import dill as pickle

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, data):
    """
    各層について、2000文ペアそれぞれのhidden_statesの類似度の平均を計算
    """
    similarities = defaultdict(list) # {layer_idx: mean_sim_of_each_sentences}

    for L1_txt, L2_txt in data:
        hidden_states = defaultdict(torch.Tensor)
        inputs_L1 = tokenizer(L1_txt, return_tensors="pt").to(device)
        inputs_L2 = tokenizer(L2_txt, return_tensors="pt").to(device)

        # get hidden_states
        with torch.no_grad():
            output_L1 = model(**inputs_L1, output_hidden_states=True)
            output_L2 = model(**inputs_L2, output_hidden_states=True)

        all_hidden_states_L1 = output_L1.hidden_states
        all_hidden_states_L2 = output_L2.hidden_states
        # 最後のtokenのhidden_statesのみ取得
        last_token_index_L1 = inputs_L1["input_ids"].shape[1] - 1
        last_token_index_L2 = inputs_L2["input_ids"].shape[1] - 1
        # 各層の最後のトークンの hidden state をリストに格納
        # last_token_hidden_states_L1 = [
        #     layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy() for layer_hidden_state in all_hidden_states_L1
        # ]
        # last_token_hidden_states_L2 = [
        #     layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy() for layer_hidden_state in all_hidden_states_L2
        # ]
        """各層の最後のトークンの hidden state をリストに格納 + 正規化 """
        last_token_hidden_states_L1 = [
            (layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy() /
            np.linalg.norm(layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy(), axis=-1, keepdims=True))
            for layer_hidden_state in all_hidden_states_L1
        ]
        last_token_hidden_states_L2 = [
            (layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy() /
            np.linalg.norm(layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy(), axis=-1, keepdims=True))
            for layer_hidden_state in all_hidden_states_L2
        ]
        # cos_sim
        similarities = calc_cosine_sim(last_token_hidden_states_L1[1:], last_token_hidden_states_L2[1:], similarities)
        # L2 distance
        # similarities = calc_euclidean_distances(last_token_hidden_states_L1, last_token_hidden_states_L2, similarities)

    return similarities

def calc_cosine_sim(last_token_hidden_states_L1: list, last_token_hidden_states_L2: list, similarities: defaultdict(float)) -> defaultdict(float):
    """
    層ごとの類似度を計算
    """
    for layer_idx, (hidden_state_L1, hidden_state_L2) in enumerate(zip(last_token_hidden_states_L1, last_token_hidden_states_L2)):
        sim = cosine_similarity(hidden_state_L1, hidden_state_L2)[0, 0] # <- [[0.50695133]] のようになっているので、数値の部分だけ抽出
        similarities[layer_idx].append(sim)

    return similarities

def calc_euclidean_distances(last_token_hidden_states_L1: list, last_token_hidden_states_L2: list, similarities: defaultdict(float)) -> defaultdict(float):
    """
    層ごとの類似度を計算
    """
    for layer_idx, (hidden_state_L1, hidden_state_L2) in enumerate(zip(last_token_hidden_states_L1, last_token_hidden_states_L2)):
        sim = euclidean_distances(hidden_state_L1, hidden_state_L2)[0, 0] # <- [[0.50695133]] のようになっているので、数値の部分だけ抽出
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
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"/home/s2410121/proj_LA/measure_similarities/llama3/images/hidden_state_sim/normalized/en_{L2}_revised.png",
        bbox_inches="tight"
        )
    plt.close()

if __name__ == "__main__":
    """ model configs """
    # LLaMA-3
    model_names = {
        # "base": "meta-llama/Meta-Llama-3-8B",
        "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
        # "de": "DiscoResearch/Llama3-German-8B", # ger
        "nl": "ReBatch/Llama-3-8B-dutch", # du
        "it": "DeepMount00/Llama-3-8b-Ita", # ita
        "ko": "beomi/Llama-3-KoEn-8B", # ko
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for L2, model_name in model_names.items():
        L1 = "en" # L1 is fixed to english.

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

        """ tatoeba translation corpus """
        dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
        # select first 2000 sentences
        total_sentence_num = 5000
        num_sentences = 2000
        dataset = dataset.select(range(total_sentence_num))
        tatoeba_data = []
        for sentence_idx, item in enumerate(dataset):
            if sentence_idx == num_sentences: break
            # check if there are empty sentences.
            if item['translation'][L1] != '' and item['translation'][L2] != '':
                tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
        # tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]
        tatoeba_data_len = len(tatoeba_data)

        """
        baseとして、対訳関係のない1文ずつのペアを作成
        """
        random_data = []
        # L1(en)
        # en_base_ds = load_dataset("agentlans/high-quality-english-sentences")
        # random_data_en = en_base_ds["train"][:num_sentences]
        # en_base_ds_idx = 0
        for sentence_idx, item in enumerate(dataset):
            if sentence_idx == num_sentences: break
            if dataset['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
                random_data.append((dataset["translation"][num_sentences+sentence_idx][L1], item["translation"][L2]))
            # en_base_ds_idx += 1

        """ calc similarities """
        results_same_semantics = calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, tatoeba_data)
        results_non_same_semantics = calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, random_data)
        final_results_same_semantics = defaultdict(float)
        final_results_non_same_semantics = defaultdict(float)
        for layer_idx in range(32): # embedding層＋３２層
            final_results_same_semantics[layer_idx] = np.array(results_same_semantics[layer_idx]).mean()
            final_results_non_same_semantics[layer_idx] = np.array(results_non_same_semantics[layer_idx]).mean()

        # print(final_results_same_semantics)
        # print(final_results_non_same_semantics)

        # delete some cache
        del model
        torch.cuda.empty_cache()

        """ plot """
        plot_hist(final_results_same_semantics, final_results_non_same_semantics, L2)
        # plot_hist_L2(final_results_same_semantics, final_results_non_same_semantics, L2)


    print("visualization completed !")
