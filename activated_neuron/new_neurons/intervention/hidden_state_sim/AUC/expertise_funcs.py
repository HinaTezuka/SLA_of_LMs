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
from typing import Any, Dict
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from baukit import Trace, TraceDict
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel
from datasets import load_dataset
from sklearn.metrics import average_precision_score

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

def track_neurons_with_text_data(model, device, model_name, tokenizer, data, is_translation_pairs: bool, activation_type: str):

    # layers_num
    num_layers = 32 if model_name == "llama" else 12
    # nums of total neurons (per a layer)
    num_neurons = 14336
    # setting pair_idx
    pair_idx = 0 if is_translation_pairs else 2000 # 0-1999: translation pairs, 2000-3999: non translation pairs

    """
    activation valuesを保存する dict (shared neuronsが対象)
    {
        pair_idx:
            layer_idx: [act_value1, act_value2, ....] <- 活性化値のリスト: idx は neuron_idx
    }
    """
    activation_dict = defaultdict(lambda: defaultdict(list))

    # Track neurons with tatoeba
    for L1_text, L2_text in data:
        """
        get activation values
        mlp_activation_L1/L2: [torch.Tensor(batch_size, sequence_length, num_neurons) * num_layers]
        """
        # L1 text
        input_ids_L1 = tokenizer(L1_text, return_tensors="pt").input_ids.to(device)
        token_len_L1 = len(input_ids_L1[0])
        act_fn_value_L1, up_proj_value_L1 = act_llama3(model, input_ids_L1)

        # L2 text
        input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to(device)
        token_len_L2 = len(input_ids_L2[0])
        act_fn_value_L2, up_proj_value_L2 = act_llama3(model, input_ids_L2)
        """
        neurons(in llama3 MLP): up_proj(x) * act_fn(gate_proj(x)) <- input of down_proj()
        L1/L2 shared neurons
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
            # neurons_L1 = torch.nonzero(torch.abs(neurons_L1_values) > active_THRESHOLD).cpu().numpy()
            # neurons_L2 = torch.nonzero(torch.abs(neurons_L2_values) > active_THRESHOLD).cpu().numpy()

            """ shared neurons: consider all neurons """
            # shared_neurons_indices = np.intersect1d(neurons_L1[:, 1], neurons_L2[:, 1])

            """ get activation_value of each shared_neurons """
            for neuron_idx in range(num_neurons):
                if activation_type == "abs":
                    act_value_L1 = get_activation_value_abs(neurons_L1_values, neuron_idx) # abs()
                    act_value_L2 = get_activation_value_abs(neurons_L2_values, neuron_idx)
                    activation_value = (act_value_L1 + act_value_L2) / 2
                elif activation_type == "product":
                    act_value_L1 = get_activation_value(neurons_L1_values, neuron_idx) # normal value
                    act_value_L2 = get_activation_value(neurons_L2_values, neuron_idx)
                    activation_value = act_value_L1 * act_value_L2
                # activation_dictに追加
                activation_dict[pair_idx][layer_idx].append((neuron_idx, activation_value))

        pair_idx += 1

    return activation_dict

def get_activation_value(activations, neuron_idx):
    """
    get activation vlaue of neuron_idx.
    """
    # 指定された層、トークン、ニューロンの発火値を取得
    activation_value = activations[0][neuron_idx].item()

    return activation_value

def get_activation_value_abs(activations, neuron_idx):
    """
    get activation vlaue of neuron_idx.
    """
    activation_value = abs(activations[0][neuron_idx].item())

    return activation_value

# def save_as_pickle(file_path, target_dict) -> None:
#     """
#     save dict as pickle file.
#     """
#     # directoryを作成（存在しない場合のみ)
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     with open(file_path, "wb") as f:
#         pickle.dump(target_dict, f)
def save_as_pickle(file_path: str, target_dict) -> None:
    """
    Save a dictionary as a pickle file with improved safety.
    """
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temp_path = file_path + ".tmp"  # Temporary file for safe writing

    try:
        # Write to a temporary file
        with open(temp_path, "wb") as f:
            pickle.dump(target_dict, f)
        # Replace the original file with the temporary file
        os.replace(temp_path, file_path)
        print("pkl_file successfully saved.")
    except Exception as e:
        # Clean up temporary file if an error occurs
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e  # Re-raise the exception for further handling

# def unfreeze_pickle(file_path: str) -> dict:
#     """
#     unfreeze pickle file as dict.
#     """
#     with open(file_path, "rb") as f:
#         return_dict = pickle.load(f)
#     return return_dict
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

""" AUC calculation setting """
def normalize_activations(activations):
    """
    Min-Max normalization.
    Args:
        activations: 活性化値のリスト
    Returns:
        正規化された活性化値のリスト
    """
    min_val = min(activations)
    max_val = max(activations)

    # 避けるべきゼロ分母問題を防ぐ
    if max_val - min_val == 0:
        print("same max and min detected.")
        return [0.0 for _ in activations]

    # Min-Max正規化適用
    return [(x - min_val) / (max_val - min_val) for x in activations]

def sigmoid_normalization(activations):
    activations = np.array(activations)  # NumPy配列に変換
    return 1 / (1 + np.exp(-activations))  # シグモイド関数を適用

def compute_ap_and_sort(label1_dict, label2_dict):
    """
    calc APscore and sort (considering nums_of_label).

    Args:
        label1_dict: activation_value for correct label(1)
        label2_dict: activation_value for incorrect label(0)

    Returns:
        ソートされたニューロンリスト, ニューロンごとの統合スコア辞書
    """
    # 各ニューロンごとの活性化値とラベルを収集
    neuron_responses = defaultdict(list)  # { (layer_idx, neuron_idx): [activation_values, ...] }
    neuron_labels = defaultdict(list)     # { (layer_idx, neuron_idx): [labels, ...] }

    # pairs for label:1
    for sentence_idx, layer_data in label1_dict.items():
        for layer_idx, neuron_activations in layer_data.items():
            for neuron_idx, activation_value in neuron_activations:
                neuron_responses[(layer_idx, neuron_idx)].append(activation_value)
                neuron_labels[(layer_idx, neuron_idx)].append(1)  # ラベル1: 正例

    # pairs for label:0
    for sentence_idx, layer_data in label2_dict.items():
        for layer_idx, neuron_activations in layer_data.items():
            for neuron_idx, activation_value in neuron_activations:
                neuron_responses[(layer_idx, neuron_idx)].append(activation_value)
                neuron_labels[(layer_idx, neuron_idx)].append(0)  # ラベル2: 負例

    # calc AP score for each shared neuron and calc total score which consider nums of label
    final_scores = {}
    for (layer_idx, neuron_idx), activations in neuron_responses.items():
        labels = neuron_labels[(layer_idx, neuron_idx)]
        # normalization
        # normalized_activations = activations

        # calc AP score
        ap = average_precision_score(y_true=labels, y_score=activations)
        # save total score
        final_scores[(layer_idx, neuron_idx)] = ap

    # sort: based on total score of each neuron
    sorted_neurons = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)

    return sorted_neurons, final_scores
