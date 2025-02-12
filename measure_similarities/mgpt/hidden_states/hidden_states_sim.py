"""
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(100000, 5120)
    (wpe): Embedding(2048, 5120)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-39): 40 x GPT2Block(
        (ln_1): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2SdpaAttention(
          (c_attn): Linear8bitLt(in_features=5120, out_features=15360, bias=True)
          (c_proj): Linear8bitLt(in_features=5120, out_features=5120, bias=True)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Linear8bitLt(in_features=5120, out_features=20480, bias=True)
          (c_proj): Linear8bitLt(in_features=20480, out_features=5120, bias=True)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=5120, out_features=100000, bias=False)
)
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
from transformers import BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, data, measure_type="cos_sim"):
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

        all_hidden_states_L1 = output_L1.hidden_states[1:] # not consider 0th_layer(=embedding_layer)
        all_hidden_states_L2 = output_L2.hidden_states[1:]
        # 最後のtokenのhidden_statesのみ取得
        last_token_index_L1 = inputs_L1["input_ids"].shape[1] - 1
        last_token_index_L2 = inputs_L2["input_ids"].shape[1] - 1

        # # 各層の最後のトークンの hidden state をリストに格納
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

        if measure_type == "cos_sim":
            # cos_sim
            similarities = calc_cosine_sim(last_token_hidden_states_L1[1:], last_token_hidden_states_L2[1:], similarities)
        if measure_type == "l2_dist":
            # L2 distance
            similarities = calc_euclidean_distances(last_token_hidden_states_L1, last_token_hidden_states_L2, similarities)

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
        f"/home/s2410121/proj_LA/measure_similarities/mgpt/images/hidden_state_sim/{L2}.png",
        bbox_inches="tight"
        )
    plt.close()

def plot_hist_L2(dict1: defaultdict(float), dict2: defaultdict(float), L2: str) -> None:
    # convert keys and values into list
    keys = np.array(list(dict1.keys()))
    # keys = list(dict1.keys())
    values1 = list(dict1.values())
    values2 = list(dict2.values())

    offset = 0.1

    # plot hist
    plt.bar(keys-offset, values1, alpha=1, label='same semantics', zorder=2)
    plt.bar(keys+offset, values2, alpha=1, label='different semantics', zorder=1)

    plt.xlabel('Layer index', fontsize=35)
    plt.ylabel('L2 dist', fontsize=35)
    plt.title(f'en_{L2}')
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"/home/s2410121/proj_LA/measure_similarities/mgpt/images/hidden_state_sim/L2_dist/{L2}.png",
        bbox_inches="tight",
    )
    plt.close()

if __name__ == "__main__":
    """ model configs """
    # mGPT
    model_name = "ai-forever/mGPT-13B"
    L2_patterns = ["ja", "nl", "ko", "it"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_16bit=True,  # 8bit quantization
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        quantization_config=quantization_config,
        device_map=device,
    )
    num_layers = 40
    L1 = "en" # L1 is fixed to english.

    for L2 in L2_patterns:
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
        baseとして、対訳関係のない1文ずつのペアを作成
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

        # """ tatoeba translation corpus """
        # dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
        # # select first 2000 sentences
        # num_sentences = 2000
        # dataset = dataset.select(range(num_sentences))
        # tatoeba_data = []
        # for item in dataset:
        #     # check if there are empty sentences.
        #     if item['translation'][L1] != '' and item['translation'][L2] != '':
        #         tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
        # # tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]
        # tatoeba_data_len = len(tatoeba_data)

        # """
        # baseとして、対訳関係のない1文ずつのペアを作成
        # (L1(en)はhttps://huggingface.co/datasets/agentlans/high-quality-english-sentences,
        # L2はtatoebaの該当データを使用)
        # """
        # random_data = []
        # # L1(en)
        # en_base_ds = load_dataset("agentlans/high-quality-english-sentences")
        # random_data_en = en_base_ds["train"][:num_sentences]
        # en_base_ds_idx = 0
        # for item in dataset:
        #     random_data.append((random_data_en["text"][en_base_ds_idx], item["translation"][L2]))
        #     en_base_ds_idx += 1

        for measure_type in ["cos_sim", "l2_dist"]:
            """ calc similarities """
            results_same_semantics = calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, tatoeba_data, measure_type)
            results_non_same_semantics = calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, random_data, measure_type)
            final_results_same_semantics = defaultdict(float)
            final_results_non_same_semantics = defaultdict(float)
            for layer_idx in range(num_layers):
                final_results_same_semantics[layer_idx] = np.array(results_same_semantics[layer_idx]).mean()
                final_results_non_same_semantics[layer_idx] = np.array(results_non_same_semantics[layer_idx]).mean()
            
            """ plot """
            if measure_type == "cos_sim":
                plot_hist(final_results_same_semantics, final_results_non_same_semantics, L2)
            elif measure_type == "l2_dist":
                plot_hist_L2(final_results_same_semantics, final_results_non_same_semantics, L2)

        # delete some cache
        torch.cuda.empty_cache()

    print("visualization completed !")
