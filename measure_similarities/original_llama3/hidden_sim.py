"""
MistralForCausalLM(
  (model): MistralModel(
    (embed_tokens): Embedding(32768, 4096)
    (layers): ModuleList(
      (0-31): 32 x MistralDecoderLayer(
        (self_attn): MistralSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): MistralRotaryEmbedding()
        )
        (mlp): MistralMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): MistralRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): MistralRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): MistralRMSNorm((4096,), eps=1e-05)
  )
  (lm_head): Linear(in_features=4096, out_features=32768, bias=False)
)
"""
import os
import sys
import pickle

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

        all_hidden_states_L1 = output_L1.hidden_states[1:]
        all_hidden_states_L2 = output_L2.hidden_states[1:]
        # 最後のtokenのhidden_statesのみ取得
        last_token_index_L1 = inputs_L1["input_ids"].shape[1] - 1
        last_token_index_L2 = inputs_L2["input_ids"].shape[1] - 1
        # 各層の最後のトークンの hidden state をリストに格納
        last_token_hidden_states_L1 = [
            layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy() for layer_hidden_state in all_hidden_states_L1
        ]
        last_token_hidden_states_L2 = [
            layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy() for layer_hidden_state in all_hidden_states_L2
        ]
        # cos_sim
        similarities = calc_cosine_sim(last_token_hidden_states_L1[1:], last_token_hidden_states_L2[1:], similarities)

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
        f"/home/s2410121/proj_LA/measure_similarities/original_llama3/images/ht_sim/{L2}.png",
        bbox_inches="tight"
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

        """ calc similarities """
        results_same_semantics = calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, tatoeba_data)
        results_non_same_semantics = calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, random_data)
        final_results_same_semantics = defaultdict(float)
        final_results_non_same_semantics = defaultdict(float)
        for layer_idx in range(32):
            final_results_same_semantics[layer_idx] = np.array(results_same_semantics[layer_idx]).mean()
            final_results_non_same_semantics[layer_idx] = np.array(results_non_same_semantics[layer_idx]).mean()

        torch.cuda.empty_cache()

        """ plot """
        plot_hist(final_results_same_semantics, final_results_non_same_semantics, L2)

    print("visualization completed !")

def compute_scores(model, tokenizer, device, data, candidate_neurons, centroids, score_type):
    """
    data: texts in specific L2 ([txt1, txt2, ...]).
    candidate_neurons: [(l, i), (l, i), ...] l: layer_idx, i: neuron_idx.
    """
    num_layers = model.config.num_hidden_layers
    final_scores = {} # { (l, i): [score1, score2, ...] }

    for text in data:
        inputs_for_ht = tokenizer(text, return_tensors="pt").to(device)
        inputs_for_att = inputs_for_ht.input_ids

        token_len = len(inputs_for_att[0])
        last_token_idx = token_len-1
        # get ht
        hts = [] # hidden_states: [ht_layer1, ht_layer2, ...]
        atts = [] # post_att_LN_outputs: [atts_layer1, atts_layer2, ...]
        acts = [] # activation_values: [acts_layer1, acts_layer2, ...]
        with torch.no_grad():
            outputs = model(**inputs_for_ht, output_hidden_states=True)
        ht_all_layer = outputs.hidden_states[1:]
        # get representation(right after post_att_LN).
        post_attention_layernorm_values = post_attention_llama3(model, inputs_for_att)
        # get activation_values(in MLP).
        act_fn_values, up_proj_values = act_llama3(model, inputs_for_att)

        for layer_idx in range(num_layers):
            # hidden states
            hts.append(ht_all_layer[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy())
            # post attention
            atts.append(post_attention_layernorm_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy())
            # act_value * up_proj
            act_fn_value = act_fn_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            up_proj_value = up_proj_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            act_values = calc_element_wise_product(act_fn_value, up_proj_value)
            acts.append(act_values)

        if score_type == "L2_dis":
            # calc scores: L2_dist((hs^l-1 + self-att() + a^l_i*v^l_i) - c) <- layerごとにあまり差が出ないので、layerの平均からの差を見る.
            c = np.mean(centroids[5:21], axis=0).reshape(1, -1)
            for layer_idx, neurons in candidate_neurons.items():
                layer_score = (hts[layer_idx-1] + atts[layer_idx]).reshape(1, -1)
                layer_score = euclidean_distances(layer_score, c)[0, 0]
                for neuron_idx in neurons:
                    value_vector = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[neuron_idx].cpu().numpy()
                    # print(value_vector)
                    score = (hts[layer_idx-1] + atts[layer_idx] + (acts[layer_idx][neuron_idx]*value_vector)).reshape(1, -1)
                    # score = value_vector.reshape(1, -1)
                    score = euclidean_distances(score, c)[0, 0]
                    score = abs(layer_score - score) if score <= layer_score else abs(layer_score - score)*-1
                    final_scores.setdefault((layer_idx, neuron_idx), []).append(score)
        elif score_type == "cos_sim":
            # calc scores: cos_sim((hs^l-1 + self-att() + a^l_i*v^l_i) - c) <- layerごとにあまり差が出ないので、layerの平均からの差を見る.
            c = np.mean(centroids[5:21], axis=0).reshape(1, -1)
            for layer_idx, neurons in candidate_neurons.items():
                layer_score = (hts[layer_idx-1] + atts[layer_idx]).reshape(1, -1)
                layer_score = cosine_similarity(layer_score, c)[0, 0]
                for neuron_idx in neurons:
                    value_vector = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[neuron_idx].cpu().numpy()
                    # print(value_vector)
                    score = (hts[layer_idx-1] + atts[layer_idx] + (acts[layer_idx][neuron_idx]*value_vector)).reshape(1, -1)
                    # score = value_vector.reshape(1, -1)
                    score = cosine_similarity(score, c)[0, 0]
                    score = abs(layer_score - score) if score >= layer_score else abs(layer_score - score)*(10**9)
                    final_scores.setdefault((layer_idx, neuron_idx), []).append(score)            

    # 
    for layer_neuron_idx, scores in final_scores.items():
        mean_score = np.mean(scores)
        final_scores[layer_neuron_idx] = mean_score

    return final_scores

def calc_score(centroids, candidate_neurons, score_type, hts, atts, acts):
    c = np.mean(centroids[5:21], axis=0).reshape(1, -1)
    layer_score = (hts[layer_idx-1] + atts[layer_idx]).reshape(1, -1)
    if score_type == "L2_dist":
        layer_score = euclidean_distances(layer_score, c)[0, 0]

    