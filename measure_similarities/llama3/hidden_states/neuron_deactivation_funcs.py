import os
import random
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import transformers
from baukit import Trace, TraceDict
from datasets import get_dataset_config_names, load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

""" func for editing activation values """
def edit_activation(output, layer, layer_idx_and_neuron_idx):
    """
    edit activation value of neurons(indexed layer_idx and neuron_idx)
    output: activation values
    layer: sth like 'model.layers.{layer_idx}.mlp.act_fn'
    layer_idx_and_neuron_idx: list of tuples like [(layer_idx, neuron_idx), ....]
    """
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if str(layer_idx) in layer:  # layer名にlayer_idxが含まれているか確認
            output[:, :, neuron_idx] *= 0  # 指定されたニューロンの活性化値をゼロに設定

    return output

def take_similarities_with_edit_activation(model, tokenizer, device, layer_neuron_list, data):
    trace_layers = [f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:

        return calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, device, data)

def calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, device, data):
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

    return similarities

def calc_cosine_sim(last_token_hidden_states_L1: list, last_token_hidden_states_L2: list, similarities: defaultdict(float)) -> defaultdict(float):
    """
    層ごとの類似度を計算
    """
    for layer_idx, (hidden_state_L1, hidden_state_L2) in enumerate(zip(last_token_hidden_states_L1, last_token_hidden_states_L2)):
        sim = cosine_similarity(hidden_state_L1, hidden_state_L2)[0, 0] # <- [[0.50695133]] のようになっているので、数値の部分だけ抽出
        similarities[layer_idx].append(sim)

    return similarities

# visualization
def plot_hist(dict1: defaultdict(float), dict2: defaultdict(float), L2: str, folder: str) -> None:
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
    plt.savefig(f"/home/s2410121/proj_LA/measure_similarities/llama3/images/hidden_state_sim_with_edit_activation/{folder}/llama3_hidden_state_sim_en_{L2}.png")
    plt.close()
