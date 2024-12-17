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
        similarities = calc_cosine_sim(last_token_hidden_states_L1, last_token_hidden_states_L2, similarities)
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
    keys = list(dict1.keys())
    values1 = list(dict1.values())
    values2 = list(dict2.values())

    # plot hist
    plt.bar(keys, values1, alpha=1, label='same semantics')
    plt.bar(keys, values2, alpha=1, label='different semantics')

    plt.xlabel('Layer index')
    plt.ylabel('Cosine Similarity')
    plt.title(f'en_{L2}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/s2410121/proj_LA/measure_similarities/llama3/images/hidden_state_sim/normalized/llama3_hidden_state_sim_en_{L2}.png")
    plt.close()

def plot_hist_L2(dict1: defaultdict(float), dict2: defaultdict(float), L2: str) -> None:
    # convert keys and values into list
    keys = list(dict1.keys())
    values1 = list(dict1.values())
    values2 = list(dict2.values())

    # plot hist
    plt.bar(keys, values1, alpha=1, label='same semantics', zorder=2)
    plt.bar(keys, values2, alpha=1, label='different semantics', zorder=1)

    plt.xlabel('Layer index')
    plt.ylabel('L2 distance')
    plt.title('L2 distance between translation pairs and non translation pairs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/s2410121/proj_LA/measure_similarities/llama3/images/hidden_state_sim/L2_dist/llama3_hidden_state_sim_en_{L2}.png")
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
        # select first 100 sentences
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

        """ calc similarities """
        results_same_semantics = calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, tatoeba_data)
        results_non_same_semantics = calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, random_data)
        final_results_same_semantics = defaultdict(float)
        final_results_non_same_semantics = defaultdict(float)
        for layer_idx in range(33): # embedding層＋３２層
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
