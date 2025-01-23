import os
import itertools
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons")
import dill as pickle
from typing import Any, Dict
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from baukit import TraceDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import average_precision_score

def get_out_gpt2(model, prompt, device, index):
    model.eval() # swith the model to evaluation mode (deactivate dropout, batch normalization)
    num_layers = model.config.num_hidden_layers  # nums of layers of the model
    MLP_act = [f"transformer.h.{i}.mlp.act" for i in range(num_layers)]  # generate path to MLP layer(of GPT-2)

    with torch.no_grad():
        # trace MLP layers using TraceDict
        with TraceDict(model, MLP_act) as ret:
            output = model(prompt, output_hidden_states=True, output_attentions=True)  # モデルを実行
        MLP_act_value = [ret[act_value].output for act_value in MLP_act]  # 各MLP層の出力を取得
        return MLP_act_value

def act_gpt2(model, input_ids):
    mlp_act = get_out_gpt2(model, input_ids, model.device, -1)  # gpt2-smallのMLP活性化を取得
    mlp_act = [act.to("cpu") for act in mlp_act] # Numpy配列はCPUでしか動かないので、各テンソルをCPU上へ移動
    # mlp_act = np.array(mlp_act)  # convert to numpy array
    # mlp_act_np = [act.detach().numpy() for act in mlp_act_tensors]
    return mlp_act

def track_neurons_with_text_data(model, device, model_name, tokenizer, data, is_translation_pairs: bool, activation_type: str):

    # layers_num
    num_layers = 32 if model_name == "llama" else 12
    # nums of total neurons (per a layer)
    num_neurons = 14336 if model_name == "llama" else 3072
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
        # print(tokenizer.decode(input_ids_L1[0][-1]))
        token_len_L1 = len(input_ids_L1[0])
        mlp_activation_L1 = act_gpt2(model, input_ids_L1)
        # L2 text
        input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to(device)
        token_len_L2 = len(input_ids_L2[0])
        mlp_activation_L2 = act_gpt2(model, input_ids_L2)

        """
        L1/L2 shared neurons
        """
        for layer_idx in range(len(mlp_activation_L1)):
            """ consider last token only """
            # L1
            mlp_activation_L1[layer_idx] = mlp_activation_L1[layer_idx][:, token_len_L1 - 1, :][0]
            # L2
            mlp_activation_L2[layer_idx] = mlp_activation_L2[layer_idx][:, token_len_L2 - 1, :][0]

            """ get activation_value of each shared_neurons """
            for neuron_idx in range(num_neurons):
                if activation_type == "abs":
                    act_value_L1 = abs(mlp_activation_L1[layer_idx][neuron_idx])
                    act_value_L2 = abs(mlp_activation_L2[layer_idx][neuron_idx])
                    activation_value = (act_value_L1 + act_value_L2) / 2
                elif activation_type == "product":
                    act_value_L1 = mlp_activation_L1[layer_idx][neuron_idx]
                    act_value_L2 = mlp_activation_L2[layer_idx][neuron_idx]
                    activation_value = act_value_L1 * act_value_L2
                # activation_dictに追加
                activation_dict[pair_idx][layer_idx].append((neuron_idx, activation_value))

        pair_idx += 1

    return activation_dict

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
