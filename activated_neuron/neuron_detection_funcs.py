"""
neuron detection for MLP Block of LLaMA-3(8B).
some codes were copied from: https://github.com/weixuan-wang123/multilingual-neurons/blob/main/neuron-behavior.ipynb
"""
import itertools
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch

from baukit import Trace, TraceDict
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel
from datasets import load_dataset

def get_out_llama3(model, prompt, device, index):
    model.eval() # swith the model to evaluation mode (deactivate dropout, batch normalization)
    num_layers = model.config.num_hidden_layers  # nums of layers of the model
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(num_layers)]  # generate path to MLP layer(of LLaMA-3)

    with torch.no_grad():
        # trace MLP layers using TraceDict
        with TraceDict(model, MLP_act) as ret:
            output = model(prompt, output_hidden_states=True, output_attentions=True)  # モデルを実行
        MLP_act_value = [ret[act_value].output for act_value in MLP_act]  # 各MLP層の出力を取得
        return MLP_act_value

def act_llama3(model, input_ids):
    mlp_act = get_out_llama3(model, input_ids, model.device, -1)  # LlamaのMLP活性化を取得
    mlp_act = [act.to("cpu") for act in mlp_act] # Numpy配列はCPUでしか動かないので、各テンソルをCPU上へ移動
    # mlp_act = np.array(mlp_act)  # convert to numpy array
    # mlp_act_np = [act.detach().numpy() for act in mlp_act_tensors]
    return mlp_act

""" shared_neuronsの発火値の合計の取得を追加 """
def track_neurons_with_text_data(model, model_name, tokenizer, data, active_THRESHOLD=0, non_active_THRESHOLD=0) -> dict:
    # Initialize lists for tracking neurons
    activated_neurons_L2 = []
    activated_neurons_L1 = []
    shared_neurons = []
    specific_neurons_L2 = []
    specific_neurons_L1 = []

    # layers_num
    num_layers = 32 if model_name == "llama" else 12
    # sentence_idx
    sentence_idx = 0

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
        # print(tokenizer.decode(input_ids_L1[0][-1]))
        token_len_L1 = len(input_ids_L1[0])
        mlp_activation_L1 = act_llama3(model, input_ids_L1)
        # L2 text
        input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to("cuda")
        token_len_L2 = len(input_ids_L2[0])
        mlp_activation_L2 = act_llama3(model, input_ids_L2)
        """
        get each type of neurons:
        L1/L2 activated neurons
        L1/L2 shared neurons
        L1/L2 non-activated neurons
        L1/L2 langage specific neurons
        """
        for layer_idx in range(len(mlp_activation_L2)):
            # Activated neurons for L1 and L2
            activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] > active_THRESHOLD).cpu().numpy()
            # condider activations only for last token
            activated_neurons_L1_layer = activated_neurons_L1_layer[activated_neurons_L1_layer[:, 1] == token_len_L1 - 1]
            activated_neurons_L1.append((layer_idx, activated_neurons_L1_layer))
            # 発火頻度の保存（layer, neuronごと）
            for neuron_idx in activated_neurons_L1_layer[:, 2]:
                act_freq_L1[layer_idx][neuron_idx] += 1
            # 発火ニューロン数の保存（プロット時に平均算出のため）
            activated_neurons_L1_vis[layer_idx].append(len(activated_neurons_L1_layer[:, 2]))

            activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] > active_THRESHOLD).cpu().numpy()
            activated_neurons_L2_layer = activated_neurons_L2_layer[activated_neurons_L2_layer[:, 1] == token_len_L2 - 1]
            activated_neurons_L2.append((layer_idx, activated_neurons_L2_layer))
            """ 発火頻度の保存（layer, neuronごと） """
            for neuron_idx in activated_neurons_L2_layer[:, 2]:
                act_freq_L2[layer_idx][neuron_idx] += 1
            # 発火ニューロン数の保存（プロット時に平均算出のため） <- 可視化用
            activated_neurons_L2_vis[layer_idx].append(len(activated_neurons_L2_layer[:, 2]))

            # Non-activated neurons for L1 and L2(shared neuronsの算出のために必要)
            non_activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
            non_activated_neurons_L1_layer = non_activated_neurons_L1_layer[non_activated_neurons_L1_layer[:, 1] == token_len_L1 - 1]

            non_activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
            non_activated_neurons_L2_layer = non_activated_neurons_L2_layer[non_activated_neurons_L2_layer[:, 1] == token_len_L2 - 1]

            # Shared neurons
            shared_neurons_layer = np.intersect1d(activated_neurons_L1_layer[:, 2], activated_neurons_L2_layer[:, 2])
            shared_neurons.append((layer_idx, shared_neurons_layer))
            """ 発火頻度の保存（layer, neuronごと） """
            for neuron_idx in shared_neurons_layer:
                act_freq_shared[layer_idx][neuron_idx] += 1
            # 発火ニューロン数の保存（プロット時に平均算出のため）
            shared_neurons_vis[layer_idx].append(len(shared_neurons_layer))
            """ L1, L2どちらかには発火はしているけど、L1/L2のshared neuronsではないneuronsを取得 (L1_actとL2_actの和集合 - L1/L2のshared neurons） """
            # L1, L2それぞれに発火しているニューロンの和集合
            union_act_L1_L2_layer = np.union1d(activated_neurons_L1_layer[:, 2], activated_neurons_L2_layer[:, 2])
            # つくった和集合から shared neuronsを取り除く
            act_L1_or_L2_neurons_layer = np.setdiff1d(union_act_L1_L2_layer, shared_neurons_layer)
            """ 発火値の累計（合計）を取得・保存(shared_neurons, act_L1_or_L2_neurons) """
            # L1/L2双方の発火値を平均して、そのlayer_idx, neuron_idxの発火値とする
            # act_sum_shared
            for neuron_idx in shared_neurons_layer:
                act_value_L1 = get_activation_value(mlp_activation_L1, layer_idx, neuron_idx, token_len_L1-1)
                act_value_L2 = get_activation_value(mlp_activation_L2, layer_idx, neuron_idx, token_len_L2-1)
                act_value = (act_value_L1 + act_value_L2) / 2
                act_sum_shared[layer_idx][neuron_idx] += act_value
            # act_sum_L1_or_L2
            for neuron_idx in act_L1_or_L2_neurons_layer:
                act_value_L1 = get_activation_value(mlp_activation_L1, layer_idx, neuron_idx, token_len_L1-1)
                act_value_L2 = get_activation_value(mlp_activation_L2, layer_idx, neuron_idx, token_len_L2-1)
                act_value = (act_value_L1 + act_value_L2) / 2
                act_sum_L1_or_L2[layer_idx][neuron_idx] += act_value
                """ 発火頻度の保存(L1 or L2) """
                act_freq_L1_or_L2[layer_idx][neuron_idx] += 1

            # Specific neurons
            specific_neurons_L1_layer = np.intersect1d(activated_neurons_L1_layer[:, 2], non_activated_neurons_L2_layer[:, 2])
            specific_neurons_L1.append((layer_idx, specific_neurons_L1_layer))
            """ 発火頻度の保存（layer, neuronごと） """
            # L1 only
            for neuron_idx in specific_neurons_L1_layer:
                act_freq_L1_only[layer_idx][neuron_idx] += 1
            # 発火ニューロン数の保存（プロット時に平均算出のため）
            specific_neurons_L1_vis[layer_idx].append(len(specific_neurons_L1_layer))

            specific_neurons_L2_layer = np.intersect1d(activated_neurons_L2_layer[:, 2], non_activated_neurons_L1_layer[:, 2])
            specific_neurons_L2.append((layer_idx, specific_neurons_L2_layer))
            """ 発火頻度の保存（layer, neuronごと） """
            # L2 only
            for neuron_idx in specific_neurons_L2_layer:
                act_freq_L2_only[layer_idx][neuron_idx] += 1
            # 発火ニューロン数の保存（プロット時に平均算出のため）
            specific_neurons_L2_vis[layer_idx].append(len(specific_neurons_L2_layer))
            """ 発火値の累計（合計）を取得・保存(L1_specific_neurons, L2_specific_neurons) """
            # L1 specific
            for neuron_idx in specific_neurons_L1_layer:
                act_value_L1 = get_activation_value(mlp_activation_L1, layer_idx, neuron_idx, token_len_L1-1)
                act_sum_L1_specific[layer_idx][neuron_idx] += act_value_L1
            # L2 specific
            for neuron_idx in specific_neurons_L2_layer:
                act_value_L2 = get_activation_value(mlp_activation_L2, layer_idx, neuron_idx, token_len_L2-1)
                act_sum_L2_specific[layer_idx][neuron_idx] += act_value_L2

        # increment sentence_idx
        sentence_idx += 1

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

def get_activation_value(mlp_activations, layer_idx, neuron_idx, token_idx):
    """
    指定された層、ニューロン、トークンの発火値を取得する関数。

    Parameters:
        mlp_activations (list): 各層のMLP活性化を含むリスト。
        layer_idx (int): 発火値を取得する層のインデックス。
        neuron_idx (int): 発火値を取得するニューロンのインデックス。
        token_idx (int): 発火値を取得するトークンのインデックス。

    Returns:
        float: 指定されたニューロンの発火値。
    """
    try:
        # 指定された層、トークン、ニューロンの発火値を取得
        activation_value = mlp_activations[layer_idx][0, token_idx, neuron_idx].item()
        return activation_value
    except IndexError:
        print(f"指定されたインデックス (layer_idx={layer_idx}, neuron_idx={neuron_idx}, token_idx={token_idx}) が範囲外です。")
        return None

def get_activation_value(mlp_activations, layer_idx, neuron_idx, token_idx):
    """
    指定されたlayer_idx、token_idx、neuron_idxの発火値を取得。

    Parameters:
        mlp_activations (list): 各層のMLP活性化を含むリスト。
        layer_idx (int): 発火値を取得する層のインデックス。
        neuron_idx (int): 発火値を取得するニューロンのインデックス。
        token_idx (int): 発火値を取得するトークンのインデックス。

    Returns:
        float: 指定されたニューロンの発火値。
    """
    activation_value = mlp_activations[layer_idx][0, token_idx, neuron_idx].item()

    return activation_value

"""
track neuron activateion
平均のプロット/発火頻度の保存バージョン
"""
# def track_neurons_with_text_data(model, model_name, tokenizer, data, active_THRESHOLD=0, non_active_THRESHOLD=0) -> dict:
#     # Initialize lists for tracking neurons
#     activated_neurons_L2 = []
#     activated_neurons_L1 = []
#     shared_neurons = []
#     specific_neurons_L2 = []
#     specific_neurons_L1 = []

#     # layers_num
#     num_layers = 32 if model_name == "llama" else 12
#     # sentence_idx
#     sentence_idx = 0

#     # 発火頻度の保存（層・ニューロンごと）: {layer_idx: neuron_idx: activation_freq}
#     act_freq_L1 = defaultdict(lambda: defaultdict(int))
#     act_freq_L2 = defaultdict(lambda: defaultdict(int))
#     act_freq_shared = defaultdict(lambda: defaultdict(int))
#     act_freq_L1_only = defaultdict(lambda: defaultdict(int))
#     act_freq_L2_only = defaultdict(lambda: defaultdict(int))

#     # layerごとの発火数の保存（平均集計・プロットのため） {layer_idx: list(all activated neurons per sentence)}
#     activated_neurons_L2_vis = defaultdict(list)
#     activated_neurons_L1_vis = defaultdict(list)
#     shared_neurons_vis = defaultdict(list)
#     specific_neurons_L2_vis = defaultdict(list)
#     specific_neurons_L1_vis = defaultdict(list)

#     # Track neurons with tatoeba
#     for L1_text, L2_text in data:
#         # L1 text
#         input_ids_L1 = tokenizer(L1_text, return_tensors="pt").input_ids.to("cuda")
#         # print(tokenizer.decode(input_ids_L1[0][-1]))
#         token_len_L1 = len(input_ids_L1[0])
#         mlp_activation_L1 = act_llama3(model, input_ids_L1)
#         # L2 text
#         input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to("cuda")
#         token_len_L2 = len(input_ids_L2[0])
#         mlp_activation_L2 = act_llama3(model, input_ids_L2)

#         for layer_idx in range(len(mlp_activation_L2)):
#             # Activated neurons for L1 and L2
#             activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] > active_THRESHOLD).cpu().numpy()
#             # 最後のトークンだけ考慮する
#             activated_neurons_L1_layer = activated_neurons_L1_layer[activated_neurons_L1_layer[:, 1] == token_len_L1 - 1]
#             activated_neurons_L1.append((layer_idx, activated_neurons_L1_layer))
#             # 発火頻度の保存（layer, neuronごと）
#             for neuron_idx in activated_neurons_L1_layer[:, 2]:
#                 act_freq_L1[layer_idx][neuron_idx] += 1
#             # 発火ニューロン数の保存（プロット時に平均算出のため）
#             activated_neurons_L1_vis[layer_idx].append(len(activated_neurons_L1_layer[:, 2]))


#             activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] > active_THRESHOLD).cpu().numpy()
#             activated_neurons_L2_layer = activated_neurons_L2_layer[activated_neurons_L2_layer[:, 1] == token_len_L2 - 1]
#             activated_neurons_L2.append((layer_idx, activated_neurons_L2_layer))
#             # 発火頻度の保存（layer, neuronごと）
#             for neuron_idx in activated_neurons_L2_layer[:, 2]:
#                 act_freq_L2[layer_idx][neuron_idx] += 1
#             # 発火ニューロン数の保存（プロット時に平均算出のため）
#             activated_neurons_L2_vis[layer_idx].append(len(activated_neurons_L2_layer[:, 2]))

#             # Non-activated neurons for L1 and L2(shared neuronsの算出のために必要)
#             non_activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
#             non_activated_neurons_L1_layer = non_activated_neurons_L1_layer[non_activated_neurons_L1_layer[:, 1] == token_len_L1 - 1]

#             non_activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
#             non_activated_neurons_L2_layer = non_activated_neurons_L2_layer[non_activated_neurons_L2_layer[:, 1] == token_len_L2 - 1]

#             # Shared neurons
#             shared_neurons_layer = np.intersect1d(activated_neurons_L1_layer[:, 2], activated_neurons_L2_layer[:, 2])
#             shared_neurons.append((layer_idx, shared_neurons_layer))
#             # 発火頻度の保存（layer, neuronごと）
#             for neuron_idx in shared_neurons_layer:
#                 act_freq_shared[layer_idx][neuron_idx] += 1
#             # 発火ニューロン数の保存（プロット時に平均算出のため）
#             shared_neurons_vis[layer_idx].append(len(shared_neurons_layer))

#             # Specific neurons
#             specific_neurons_L1_layer = np.intersect1d(activated_neurons_L1_layer[:, 2], non_activated_neurons_L2_layer[:, 2])
#             specific_neurons_L1.append((layer_idx, specific_neurons_L1_layer))
#             # 発火頻度の保存（layer, neuronごと）
#             for neuron_idx in specific_neurons_L1_layer:
#                 act_freq_L1_only[layer_idx][neuron_idx] += 1
#             # 発火ニューロン数の保存（プロット時に平均算出のため）
#             specific_neurons_L1_vis[layer_idx].append(len(specific_neurons_L1_layer))

#             specific_neurons_L2_layer = np.intersect1d(activated_neurons_L2_layer[:, 2], non_activated_neurons_L1_layer[:, 2])
#             specific_neurons_L2.append((layer_idx, specific_neurons_L2_layer))
#             # 発火頻度の保存（layer, neuronごと）
#             for neuron_idx in specific_neurons_L2_layer:
#                 act_freq_L2_only[layer_idx][neuron_idx] += 1
#             # 発火ニューロン数の保存（プロット時に平均算出のため）
#             specific_neurons_L2_vis[layer_idx].append(len(specific_neurons_L2_layer))

#         # increment sentence_idx
#         sentence_idx += 1

#     # Create output dictionaries
#     output_dict = {
#         "activated_neurons_L1": activated_neurons_L1,
#         "activated_neurons_L2": activated_neurons_L2,
#         "shared_neurons": shared_neurons,
#         "specific_neurons_L1": specific_neurons_L1,
#         "specific_neurons_L2": specific_neurons_L2,
#     }

#     # 各文ペア、各層、各ニューロンの発火ニューロン数
#     output_dict_vis = {
#         "activated_neurons_L1": activated_neurons_L1_vis,
#         "activated_neurons_L2": activated_neurons_L2_vis,
#         "shared_neurons": shared_neurons_vis,
#         "specific_neurons_L1": specific_neurons_L1_vis,
#         "specific_neurons_L2": specific_neurons_L2_vis,
#     }

#     # 各層の各ニューロンごとの発火頻度
#     freq_dict = {
#         # 発火頻度の保存（層・ニューロンごと）: {layer_idx: neuron_idx: activation_freq}
#         "activated_neurons_L1": act_freq_L1,
#         "activated_neurons_L2": act_freq_L2,
#         "shared_neurons": act_freq_shared,
#         "specific_neurons_L1": act_freq_L1_only,
#         "specific_neurons_L2": act_freq_L2_only,
#     }

#     return output_dict, output_dict_vis, freq_dict

""" track neuron activateion(合計ver.) """
# def track_neurons_with_text_data(model, model_name, tokenizer, data, active_THRESHOLD=0, non_active_THRESHOLD=0) -> dict:
#     # Initialize lists for tracking neurons
#     activated_neurons_L2 = []
#     activated_neurons_L1 = []
#     non_activated_neurons_L2 = []
#     non_activated_neurons_L1 = []
#     shared_neurons = []
#     specific_neurons_L2 = []
#     specific_neurons_L1 = []
#     non_activated_neurons_all = []

#     # Initialize sets for visualization
#     num_layers = 32 if model_name == "llama" else 12
#     activated_neurons_L2_vis = [set() for _ in range(num_layers)]
#     activated_neurons_L1_vis = [set() for _ in range(num_layers)]
#     non_activated_neurons_L2_vis = [set() for _ in range(num_layers)]
#     non_activated_neurons_L1_vis = [set() for _ in range(num_layers)]
#     shared_neurons_vis = [set() for _ in range(num_layers)]
#     specific_neurons_L2_vis = [set() for _ in range(num_layers)]
#     specific_neurons_L1_vis = [set() for _ in range(num_layers)]
#     non_activated_neurons_all_vis = [set() for _ in range(num_layers)]

#     # dict for tracking layers and shared_neurons for analysis later
#     track_dict = defaultdict(set) # layer_idx: neuron_indices(set)

#     # Track neurons with tatoeba
#     for L1_text, L2_text in data:
#         # L1 text
#         input_ids_L1 = tokenizer(L1_text, return_tensors="pt").input_ids.to("cuda")
#         # print(tokenizer.decode(input_ids_L1[0][-1]))
#         token_len_L1 = len(input_ids_L1[0])
#         mlp_activation_L1 = act_llama3(model, input_ids_L1)
#         # L2 text
#         input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to("cuda")
#         token_len_L2 = len(input_ids_L2[0])
#         mlp_activation_L2 = act_llama3(model, input_ids_L2)

#         for layer_idx in range(len(mlp_activation_L2)):
#             # Activated neurons for L1 and L2
#             activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] > active_THRESHOLD).cpu().numpy()
#             # 最初の文頭トークンをカウントしない
#             # activated_neurons_L1_layer = activated_neurons_L1_layer[activated_neurons_L1_layer[:, 1] != 0]
#             # 最後のトークンだけ考慮する
#             activated_neurons_L1_layer = activated_neurons_L1_layer[activated_neurons_L1_layer[:, 1] == token_len_L1 - 1]
#             activated_neurons_L1.append((layer_idx, activated_neurons_L1_layer))
#             activated_neurons_L1_vis[layer_idx].update(np.unique(activated_neurons_L1_layer[:, 2]))

#             activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] > active_THRESHOLD).cpu().numpy()
#             # activated_neurons_L2_layer = activated_neurons_L2_layer[activated_neurons_L2_layer[:, 1] != 0]
#             activated_neurons_L2_layer = activated_neurons_L2_layer[activated_neurons_L2_layer[:, 1] == token_len_L2 - 1]
#             activated_neurons_L2.append((layer_idx, activated_neurons_L2_layer))
#             activated_neurons_L2_vis[layer_idx].update(np.unique(activated_neurons_L2_layer[:, 2]))

#             # Non-activated neurons for L1 and L2
#             non_activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
#             # non_activated_neurons_L1_layer = non_activated_neurons_L1_layer[non_activated_neurons_L1_layer[:, 1] != 0]
#             non_activated_neurons_L1_layer = non_activated_neurons_L1_layer[non_activated_neurons_L1_layer[:, 1] == token_len_L1 - 1]
#             non_activated_neurons_L1.append((layer_idx, non_activated_neurons_L1_layer))
#             non_activated_neurons_L1_vis[layer_idx].update(np.unique(non_activated_neurons_L1_layer[:, 2]))

#             non_activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
#             # non_activated_neurons_L2_layer = non_activated_neurons_L2_layer[non_activated_neurons_L2_layer[:, 1] != 0]
#             non_activated_neurons_L2_layer = non_activated_neurons_L2_layer[non_activated_neurons_L2_layer[:, 1] == token_len_L2 - 1]
#             non_activated_neurons_L2.append((layer_idx, non_activated_neurons_L2_layer))
#             non_activated_neurons_L2_vis[layer_idx].update(np.unique(non_activated_neurons_L2_layer[:, 2]))

#             # Non-activated neurons for both L1 and L2
#             non_activated_neurons_both_L1_L2_layer = np.intersect1d(non_activated_neurons_L1_layer[:, 2], non_activated_neurons_L2_layer[:, 2])
#             non_activated_neurons_all.append((layer_idx, non_activated_neurons_both_L1_L2_layer))
#             non_activated_neurons_all_vis[layer_idx].update(non_activated_neurons_both_L1_L2_layer)

#             # Shared neurons
#             shared_neurons_layer = np.intersect1d(activated_neurons_L1_layer[:, 2], activated_neurons_L2_layer[:, 2])
#             shared_neurons.append((layer_idx, shared_neurons_layer))
#             shared_neurons_vis[layer_idx].update(shared_neurons_layer)
#             """ track layer_idx and neuron_idx for analysis """
#             track_dict = extract_neuron_indices(track_dict, layer_idx, shared_neurons_layer)

#             # Specific neurons
#             specific_neurons_L1_layer = np.intersect1d(activated_neurons_L1_layer[:, 2], non_activated_neurons_L2_layer[:, 2])
#             specific_neurons_L1.append((layer_idx, specific_neurons_L1_layer))
#             specific_neurons_L1_vis[layer_idx].update(specific_neurons_L1_layer)

#             specific_neurons_L2_layer = np.intersect1d(activated_neurons_L2_layer[:, 2], non_activated_neurons_L1_layer[:, 2])
#             specific_neurons_L2.append((layer_idx, specific_neurons_L2_layer))
#             specific_neurons_L2_vis[layer_idx].update(specific_neurons_L2_layer)

#     # Create output dictionaries
#     output_dict = {
#         "activated_neurons_L1": activated_neurons_L1,
#         "activated_neurons_L2": activated_neurons_L2,
#         "non_activated_neurons_L1": non_activated_neurons_L1,
#         "non_activated_neurons_L2": non_activated_neurons_L2,
#         "shared_neurons": shared_neurons,
#         "specific_neurons_L1": specific_neurons_L1,
#         "specific_neurons_L2": specific_neurons_L2,
#         "non_activated_neurons_all": non_activated_neurons_all
#     }

#     output_dict_vis = {
#         "activated_neurons_L1": activated_neurons_L1_vis,
#         "activated_neurons_L2": activated_neurons_L2_vis,
#         "non_activated_neurons_L1": non_activated_neurons_L1_vis,
#         "non_activated_neurons_L2": non_activated_neurons_L2_vis,
#         "shared_neurons": shared_neurons_vis,
#         "specific_neurons_L1": specific_neurons_L1_vis,
#         "specific_neurons_L2": specific_neurons_L2_vis,
#         "non_activated_neurons_all": non_activated_neurons_all_vis
#     }

#     return output_dict, output_dict_vis, track_dict

""" 発火したニューロンのindexなども追跡 """
def extract_neuron_indices(track_dict: defaultdict(set), layer_idx: int, activation_layer: np.array):
    """
    make a dict for tracking which neuron of which layer are activated/non-activated as idx
    (both for shared neurons and language specific neurons)

    activation_layer: np.array which has indices of activated neurons
    """
    track_dict[layer_idx].update(np.unique(activation_layer))

    return track_dict


""" sentenceごとの管理を追加 """
# def track_neurons_with_text_data(model, tokenizer, data, active_THRESHOLD=0, non_active_THRESHOLD=0) -> dict: # data <- 通常はtatoeba
#    # list for tracking neurons: len(data) = nums of sentence
#     activated_neurons_L2 = [[] for _ in range(len(data))]
#     activated_neurons_L1 = [[] for _ in range(len(data))]
#     non_activated_neurons_L2 = [[] for _ in range(len(data))]
#     non_activated_neurons_L1 = [[] for _ in range(len(data))]
#     shared_neurons = [[] for _ in range(len(data))]
#     specific_neurons_L2 = [[] for _ in range(len(data))]
#     specific_neurons_L1 = [[] for _ in range(len(data))]
#     non_activated_neurons_all = [[] for _ in range(len(data))]

#     sentence_idx = 0
#     """ track neurons with tatoeba """
#     for L1_text, L2_text in data:
#         # L1 text
#         input_ids_L1 = tokenizer(L1_text, return_tensors="pt").input_ids.to("cuda")
#         mlp_activation_L1 = act_llama3(model, input_ids_L1)
#         # L2 text
#         input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to("cuda")
#         mlp_activation_L2 = act_llama3(model, input_ids_L2)
#         """
#         mlp_activation_L1/L2 <- 層の数（llamaなら32）のリスト
#         len(mlp_activation_L1): nums of layer (32: llamaの場合)
#         len(mlp_activation_L1[0]): batch size <- 1(1文しか入れていないため)
#         len(mlp_activation_L1[0][0]): nums of token(of a input sentence)
#         len(mlp_activation_L1[0][0][0]): 14336(= nums of neuron)
#         """

#         """ aggregate each type of neurons per each layer """
#         for layer_idx in range(len(mlp_activation_L2)):

#             """ activated neurons """
#             # activated neurons for L1
#             activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] > active_THRESHOLD).cpu().numpy()
#             # remove activation to 0-th token (for LLaMA3, it's <|begin_of_text|>)
#             activated_neurons_L1_layer = activated_neurons_L1_layer[activated_neurons_L1_layer[:, 1] != 0]
#             activated_neurons_L1[sentence_idx].append((layer_idx, activated_neurons_L1_layer))
#             # activated neurons for L2
#             activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] > active_THRESHOLD).cpu().numpy()
#             # remove activation to 0-th token (for LLaMA3, it's <|begin_of_text|>)
#             activated_neurons_L2_layer = activated_neurons_L2_layer[activated_neurons_L2_layer[:, 1] != 0]
#             activated_neurons_L2[sentence_idx].append((layer_idx, activated_neurons_L2_layer))

#             """ non-activated neurons """
#             # non-activated neurons for L1
#             non_activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
#             non_activated_neurons_L1_layer = non_activated_neurons_L1_layer[non_activated_neurons_L1_layer[:, 1] != 0]
#             non_activated_neurons_L1[sentence_idx].append((layer_idx, non_activated_neurons_L1_layer))
#             # non-activated neurons for L2
#             non_activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
#             non_activated_neurons_L2_layer = non_activated_neurons_L2_layer[non_activated_neurons_L2_layer[:, 1] != 0]
#             non_activated_neurons_L2[sentence_idx].append((layer_idx, non_activated_neurons_L2_layer))
#             # non-activated_neurons for both L1 and L2
#             non_activated_neurons_both_L1_L2_layer = np.intersect1d(non_activated_neurons_L1_layer[:, 2], non_activated_neurons_L2_layer[:, 2])
#             non_activated_neurons_all[sentence_idx].append((layer_idx, non_activated_neurons_both_L1_L2_layer))

#             """ shared neurons """
#             # shared neurons for both L1 and L2
#             shared_neurons_layer = np.intersect1d(activated_neurons_L1_layer[:, 2], activated_neurons_L2_layer[:, 2])
#             # shared_neurons_layer = np.intersect1d(activated_neurons_L2_layer, activated_neurons_L1_layer)
#             shared_neurons[sentence_idx].append((layer_idx, shared_neurons_layer))

#             """ specific neurons """
#             specific_neurons_L1_layer = np.intersect1d(activated_neurons_L1_layer[:, 2], non_activated_neurons_L2_layer[:, 2])
#             # specific_neurons_L1_layer = np.intersect1d(specific_neurons_L1_layer, shared_neurons_layer)
#             specific_neurons_L1[sentence_idx].append((layer_idx, specific_neurons_L1_layer))
#             # specific neurons for L2
#             specific_neurons_L2_layer = np.intersect1d(activated_neurons_L2_layer[:, 2], non_activated_neurons_L1_layer[:, 2])
#             # specific_neurons_L2_layer = np.intersect1d(specific_neurons_L2_layer, shared_neurons_layer)
#             specific_neurons_L2[sentence_idx].append((layer_idx, specific_neurons_L2_layer))

#         sentence_idx += 1

#     output_dict = {
#         "activated_neurons_L1": activated_neurons_L1,
#         "activated_neurons_L2": activated_neurons_L2,
#         "non_activated_neurons_L1": non_activated_neurons_L1,
#         "non_activated_neurons_L2": non_activated_neurons_L2,
#         "shared_neurons": shared_neurons,
#         "specific_neurons_L1": specific_neurons_L1,
#         "specific_neurons_L2": specific_neurons_L2,
#         "non_activated_neurons_all": non_activated_neurons_all
#     }

#     return output_dict
