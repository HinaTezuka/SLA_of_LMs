"""
neuron detection for MLP Block of LLaMA-3(8B).
some codes were copied from: https://github.com/weixuan-wang123/multilingual-neurons/blob/main/neuron-behavior.ipynb
"""
"""
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
"""
import os
import itertools
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")
import dill as pickle

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch

from baukit import Trace, TraceDict
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel
from datasets import load_dataset

def get_out_llama3_act_fn(model, prompt, device, index):
    model.eval() # swith the model to evaluation mode (deactivate dropout, batch normalization)
    num_layers = model.config.num_hidden_layers  # nums of layers of the model
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(num_layers)]  # generate path to MLP layer(of LLaMA-3)

    with torch.no_grad():
        # trace MLP layers using TraceDict
        with TraceDict(model, MLP_act) as ret:
            output = model(prompt, output_hidden_states=True, output_attentions=True)  # モデルを実行
        MLP_act_value = [ret[act_value].output for act_value in MLP_act]  # 各MLP層の出力を取得
        return MLP_act_value

def get_out_llama3_up_proj(model, prompt, device, index):
    model.eval() # swith the model to evaluation mode (deactivate dropout, batch normalization)
    num_layers = model.config.num_hidden_layers  # nums of layers of the model
    MLP_act = [f"model.layers.{i}.mlp.up_proj" for i in range(num_layers)]  # generate path to MLP layer(of LLaMA-3)

    with torch.no_grad():
        # trace MLP layers using TraceDict
        with TraceDict(model, MLP_act) as ret:
            output = model(prompt, output_hidden_states=True, output_attentions=True)
        MLP_act_value = [ret[act_value].output for act_value in MLP_act]
        return MLP_act_value

def act_llama3(model, input_ids):
    act_fn_values = get_out_llama3_act_fn(model, input_ids, model.device, -1)  # LlamaのMLP活性化を取得
    act_fn_values = [act.to("cpu") for act in act_fn_values] # Numpy配列はCPUでしか動かないので、各テンソルをCPU上へ移動
    up_proj_values = get_out_llama3_up_proj(model, input_ids, model.device, -1)
    up_proj_values = [act.to("cpu") for act in up_proj_values]

    return act_fn_values, up_proj_values

# act_fn(x)とup_proj(x)の要素積を計算(=neuronとする)
def calc_element_wise_product(act_fn_value, up_proj_value):
    return act_fn_value * up_proj_value

""" shared_neuronsの発火値の合計の取得を追加 """
def track_neurons_with_text_data(model, model_name, tokenizer, data, active_THRESHOLD=0.01, non_active_THRESHOLD=0):
    # Initialize lists for tracking neurons
    activated_neurons_L2 = []
    activated_neurons_L1 = []
    shared_neurons = []
    specific_neurons_L2 = []
    specific_neurons_L1 = []

    # layers_num
    num_layers = 32 if model_name == "llama" else 12

    # 発火頻度の保存（層・ニューロンごと）: {layer_idx: neuron_idx: activation_freq}
    act_freq_L1 = defaultdict(lambda: defaultdict(int))
    act_freq_L2 = defaultdict(lambda: defaultdict(int))
    act_freq_shared = defaultdict(lambda: defaultdict(int))
    act_freq_L1_or_L2 = defaultdict(lambda: defaultdict(int))
    act_freq_L1_only = defaultdict(lambda: defaultdict(int))
    act_freq_L2_only = defaultdict(lambda: defaultdict(int))

    # layerごとの発火数の保存（平均集計・プロットのため） {layer_idx: list(all activated neurons per sentence)}
    activated_neurons_L2_vis = defaultdict(list)
    activated_neurons_L1_vis = defaultdict(list)
    shared_neurons_vis = defaultdict(list)
    specific_neurons_L2_vis = defaultdict(list)
    specific_neurons_L1_vis = defaultdict(list)

    # 発火値の合計の保存（層・ニューロンごと）: {layer_idx: neuron_idx: activation_values_sum}
    act_sum_shared = defaultdict(lambda: defaultdict(float))
    act_sum_L1_or_L2 = defaultdict(lambda: defaultdict(float))
    act_sum_L1_specific = defaultdict(lambda: defaultdict(float))
    act_sum_L2_specific = defaultdict(lambda: defaultdict(float))

    # Track neurons with tatoeba
    for L1_text, L2_text in data:
        """
        get activation values
        mlp_activation_L1/L2: [torch.Tensor(batch_size, sequence_length, num_neurons) * num_layers]
        """
        # L1 text
        input_ids_L1 = tokenizer(L1_text, return_tensors="pt").input_ids.to("cuda")
        token_len_L1 = len(input_ids_L1[0])
        act_fn_value_L1, up_proj_value_L1 = act_llama3(model, input_ids_L1)

        # L2 text
        input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to("cuda")
        token_len_L2 = len(input_ids_L2[0])
        act_fn_value_L2, up_proj_value_L2 = act_llama3(model, input_ids_L2)
        """
        neurons(in llama3 MLP): up_proj(x) * act_fn(x)
        get each type of neurons:
        L1/L2 activated neurons
        L1/L2 shared neurons
        L1/L2 non-activated neurons
        L1/L2 langage specific neurons
        """
        for layer_idx in range(len(act_fn_value_L1)):
            """ consider last token only """
            # L1
            act_fn_value_L1[layer_idx] = act_fn_value_L1[layer_idx][:, token_len_L1 - 1, :]
            up_proj_value_L1[layer_idx] = up_proj_value_L1[layer_idx][:, token_len_L1 - 1, :]
            # L2
            act_fn_value_L2[layer_idx] = act_fn_value_L2[layer_idx][:, token_len_L2 - 1, :]
            up_proj_value_L2[layer_idx] = up_proj_value_L2[layer_idx][:, token_len_L2 - 1, :]
            """ calc and extract neurons: up_proj(x) * act_fn(x) """
            neurons_L1_values = calc_element_wise_product(act_fn_value_L1[layer_idx], up_proj_value_L1[layer_idx]) # torch.Tensor
            neurons_L2_values = calc_element_wise_product(act_fn_value_L2[layer_idx], up_proj_value_L2[layer_idx])
            """ calc abs_values of each activation_values and sort """
            # 要素ごとの絶対値が active_THRESHOLD を超えている場合のインデックスを取得
            neurons_L1 = torch.nonzero(torch.abs(neurons_L1_values) > active_THRESHOLD).cpu().numpy()
            neurons_L2 = torch.nonzero(torch.abs(neurons_L2_values) > active_THRESHOLD).cpu().numpy()
            #
            activated_neurons_L1.append((layer_idx, neurons_L1))
            activated_neurons_L2.append((layer_idx, neurons_L2))
            # 発火頻度の保存（layer, neuronごと）
            for neuron_idx in neurons_L1[:, 1]:
                act_freq_L1[layer_idx][neuron_idx] += 1
            for neuron_idx in neurons_L2[:, 1]:
                act_freq_L2[layer_idx][neuron_idx] += 1
            # 発火ニューロン数の保存（プロット時に平均算出のため）
            activated_neurons_L1_vis[layer_idx].append(len(neurons_L1[:, 1]))
            activated_neurons_L2_vis[layer_idx].append(len(neurons_L2[:, 1]))

            """ shared neurons """
            shared_neurons_indices = np.intersect1d(neurons_L1[:, 1], neurons_L2[:, 1])
            shared_neurons.append((layer_idx, shared_neurons_indices))

            # 発火値の累計（合計） / 発火頻度を保存(shared_neurons)
            for neuron_idx in shared_neurons_indices:
                # act_sum_shared
                act_value_L1 = get_activation_value(neurons_L1_values, neuron_idx)
                act_value_L2 = get_activation_value(neurons_L2_values, neuron_idx)
                act_value = act_value_L1 + act_value_L2
                act_sum_shared[layer_idx][neuron_idx] += act_value
                # act_freq_shared
                act_freq_shared[layer_idx][neuron_idx] += 1
            # 発火ニューロン数の保存（プロット時に平均算出のため）
            shared_neurons_vis[layer_idx].append(len(shared_neurons_indices))

            """ L1, L2どちらかには発火はしているけど、L1/L2のshared neuronsではないneuronsを取得 (L1_actとL2_actの和集合 - L1/L2のshared neurons） """
            # L1, L2それぞれに発火しているニューロンの和集合
            union_act_L1_L2_layer = np.union1d(neurons_L1[:, 1], neurons_L2[:, 1])
            # つくった和集合から shared neuronsを取り除く = L1 or L2に発火している neurons
            act_L1_or_L2_neurons = np.setdiff1d(union_act_L1_L2_layer, shared_neurons_indices)

            # 発火値の累計（合計） / 発火頻度を保存(act_L1_or_L2_neurons)
            for neuron_idx in act_L1_or_L2_neurons:
                # act_sum_shared
                act_value_L1 = get_activation_value(neurons_L1_values, neuron_idx)
                act_value_L2 = get_activation_value(neurons_L2_values, neuron_idx)
                act_value = act_value_L1 + act_value_L2
                act_sum_L1_or_L2[layer_idx][neuron_idx] += act_value
                # act_freq_shared
                act_freq_L1_or_L2[layer_idx][neuron_idx] += 1

            """ non-activated neurons for L1 / L2: <- specific neuronsの検出のために必要 """
            non_activated_neurons_L1 = torch.nonzero(torch.abs(neurons_L1_values) <= active_THRESHOLD).cpu().numpy()
            non_activated_neurons_L2 = torch.nonzero(torch.abs(neurons_L2_values) <= active_THRESHOLD).cpu().numpy()

            """ Specific neurons """
            # L1
            specific_neurons_L1_indices = np.intersect1d(neurons_L1[:, 1], non_activated_neurons_L2)
            specific_neurons_L1.append((layer_idx, specific_neurons_L1_indices))
            # L2
            specific_neurons_L2_indices = np.intersect1d(neurons_L2[:, 1], non_activated_neurons_L1)
            specific_neurons_L2.append((layer_idx, specific_neurons_L2_indices))
            """ 発火頻度/発火値の合計の保存（layer, neuronごと） """
            # L1 specific
            for neuron_idx in specific_neurons_L1_indices:
                # act_freq
                act_freq_L1_only[layer_idx][neuron_idx] += 1 # L1
                # act_sum
                act_value_L1 = get_activation_value(neurons_L1_values, neuron_idx)
                act_sum_L1_specific[layer_idx][neuron_idx] += act_value_L1
            # L2 specific
            for neuron_idx in specific_neurons_L2_indices:
                # act_freq
                act_freq_L2_only[layer_idx][neuron_idx] += 1 # L2
                # act_sum
                act_value_L2 = get_activation_value(neurons_L2_values, neuron_idx)
                act_sum_L2_specific[layer_idx][neuron_idx] += act_value_L2

            # 発火ニューロン数の保存（プロット時に平均算出のため）: L1 / L2 specific
            specific_neurons_L1_vis[layer_idx].append(len(specific_neurons_L1_indices))
            specific_neurons_L2_vis[layer_idx].append(len(specific_neurons_L2_indices))

    # Create output dictionaries
    output_dict = {
        "activated_neurons_L1": activated_neurons_L1,
        "activated_neurons_L2": activated_neurons_L2,
        "shared_neurons": shared_neurons,
        "specific_neurons_L1": specific_neurons_L1,
        "specific_neurons_L2": specific_neurons_L2,
    }
    # 各文ペア、各層、各ニューロンの発火ニューロン数
    output_dict_vis = {
        "activated_neurons_L1": activated_neurons_L1_vis,
        "activated_neurons_L2": activated_neurons_L2_vis,
        "shared_neurons": shared_neurons_vis,
        "specific_neurons_L1": specific_neurons_L1_vis,
        "specific_neurons_L2": specific_neurons_L2_vis,
    }
    # 各層の各ニューロンごとの発火頻度
    freq_dict = {
        # 発火頻度の保存（層・ニューロンごと）: {layer_idx: neuron_idx: activation_freq}
        "activated_neurons_L1": act_freq_L1,
        "activated_neurons_L2": act_freq_L2,
        "shared_neurons": act_freq_shared,
        "activated_neurons_L1_or_L2": act_freq_L1_or_L2,
        "specific_neurons_L1": act_freq_L1_only,
        "specific_neurons_L2": act_freq_L2_only,
    }
    # 各層の各ニューロンごとの発火値合計
    act_sum_dict = {
        "shared": act_sum_shared,
        "L1_or_L2": act_sum_L1_or_L2,
        "L1_specific": act_sum_L1_specific,
        "L2_specific": act_sum_L2_specific,
    }

    return output_dict, output_dict_vis, freq_dict, act_sum_dict

def get_activation_value(activations, neuron_idx):
    """
    get activation vlaue of neuron_idx.
    """
    # 指定された層、トークン、ニューロンの発火値を取得
    activation_value = abs(activations[0][neuron_idx].item())

    return activation_value

def save_as_pickle(file_path, target_dict) -> None:
    """
    save dict as pickle file.
    """
    # directoryを作成（存在しない場合のみ)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(target_dict, f)
