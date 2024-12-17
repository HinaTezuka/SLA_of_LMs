"""
neuron detection for MLP Block of LLaMA-3(8B).
some codes were copied from: https://github.com/weixuan-wang123/multilingual-neurons/blob/main/neuron-behavior.ipynb
"""
import itertools
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")

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

model_name = "tokyotech-llm/Llama-3-Swallow-8B-v0.1"
text = "お腹がすいた"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
print(input_ids)
sys.exit()


def track_neurons_with_text_data_last_token_only(model, model_name, tokenizer, data, active_THRESHOLD=0, non_active_THRESHOLD=0) -> dict: # data <- 通常はtatoeba
   # list for tracking neurons: len(data) = nums of sentence
    activated_neurons_L2 = []
    activated_neurons_L1 = []
    non_activated_neurons_L2 = []
    non_activated_neurons_L1 = []
    shared_neurons = []
    specific_neurons_L2 = []
    specific_neurons_L1 = []
    non_activated_neurons_all = []
    """ for visualization """
    activated_neurons_L2_vis = [[] for _ in range(32)]
    activated_neurons_L1_vis = [[] for _ in range(32)]
    non_activated_neurons_L2_vis = [[] for _ in range(32)]
    non_activated_neurons_L1_vis = [[] for _ in range(32)]
    shared_neurons_vis = [[] for _ in range(32)]
    specific_neurons_L2_vis = [[] for _ in range(32)]
    specific_neurons_L1_vis = [[] for _ in range(32)]
    non_activated_neurons_all_vis = [[] for _ in range(32)]

    num_layers = 32 if model_name == "llama" else 12
    """ track neurons with tatoeba """
    for L1_text, L2_text in data:
        # L1 text
        input_ids_L1 = tokenizer(L1_text, return_tensors="pt").input_ids.to("cuda")
        mlp_activation_L1 = act_llama3(model, input_ids_L1)
        # L2 text
        input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to("cuda")
        mlp_activation_L2 = act_llama3(model, input_ids_L2)
        """
        mlp_activation_L1/L2 <- 層の数（llamaなら32）のリスト
        len(mlp_activation_L1): nums of layer (32: llamaの場合)
        len(mlp_activation_L1[0]): batch size <- 1(1文しか入れていないため)
        len(mlp_activation_L1[0][0]): nums of token(of a input sentence)
        len(mlp_activation_L1[0][0][0]): 14336(= nums of neuron)
        """

        """ aggregate each type of neurons per each layer """
        for layer_idx in range(len(mlp_activation_L2)):

            """ activated neurons """
            # activated neurons for L1
            activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] > active_THRESHOLD).cpu().numpy()
            # remove activation to 0-th token (for LLaMA3, it's <|begin_of_text|>)
            activated_neurons_L1_layer = activated_neurons_L1_layer[activated_neurons_L1_layer[:, 1] != 0]
            activated_neurons_L1.append((layer_idx, activated_neurons_L1_layer))
            activated_neurons_L1_vis[layer_idx].append(np.unique(activated_neurons_L1_layer[:, 2]).tolist())
            # activated neurons for L2
            activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] > active_THRESHOLD).cpu().numpy()
            # remove activation to 0-th token (for LLaMA3, it's <|begin_of_text|>)
            activated_neurons_L2_layer = activated_neurons_L2_layer[activated_neurons_L2_layer[:, 1] != 0]
            activated_neurons_L2.append((layer_idx, activated_neurons_L2_layer))
            activated_neurons_L2_vis[layer_idx].append(np.unique(activated_neurons_L2_layer[:, 2]).tolist())

            """ non-activated neurons """
            # non-activated neurons for L1
            non_activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
            non_activated_neurons_L1_layer = non_activated_neurons_L1_layer[non_activated_neurons_L1_layer[:, 1] != 0]
            non_activated_neurons_L1.append((layer_idx, non_activated_neurons_L1_layer))
            non_activated_neurons_L1_vis[layer_idx].append(np.unique(non_activated_neurons_L1_layer[:, 2]).tolist())
            # non-activated neurons for L2
            non_activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
            non_activated_neurons_L2_layer = non_activated_neurons_L2_layer[non_activated_neurons_L2_layer[:, 1] != 0]
            non_activated_neurons_L2.append((layer_idx, non_activated_neurons_L2_layer))
            non_activated_neurons_L2_vis[layer_idx].append(np.unique(non_activated_neurons_L2_layer[:, 2]).tolist())
            # non-activated_neurons for both L1 and L2
            non_activated_neurons_both_L1_L2_layer = np.intersect1d(non_activated_neurons_L1_layer[:, 2], non_activated_neurons_L2_layer[:, 2])
            non_activated_neurons_all.append((layer_idx, non_activated_neurons_both_L1_L2_layer))
            non_activated_neurons_all_vis[layer_idx].append(non_activated_neurons_both_L1_L2_layer)

            """ shared neurons """
            # shared neurons for both L1 and L2
            shared_neurons_layer = np.intersect1d(activated_neurons_L1_layer[:, 2], activated_neurons_L2_layer[:, 2])
            # shared_neurons_layer = np.intersect1d(activated_neurons_L2_layer, activated_neurons_L1_layer)
            shared_neurons.append((layer_idx, shared_neurons_layer))
            shared_neurons_vis[layer_idx].append(shared_neurons_layer)

            """ specific neurons """
            specific_neurons_L1_layer = np.intersect1d(activated_neurons_L1_layer[:, 2], non_activated_neurons_L2_layer[:, 2])
            # specific_neurons_L1_layer = np.intersect1d(specific_neurons_L1_layer, shared_neurons_layer)
            specific_neurons_L1.append((layer_idx, specific_neurons_L1_layer))
            specific_neurons_L1_vis[layer_idx].append(specific_neurons_L1_layer)
            # specific neurons for L2
            specific_neurons_L2_layer = np.intersect1d(activated_neurons_L2_layer[:, 2], non_activated_neurons_L1_layer[:, 2])
            # specific_neurons_L2_layer = np.intersect1d(specific_neurons_L2_layer, shared_neurons_layer)
            specific_neurons_L2.append((layer_idx, specific_neurons_L2_layer))
            specific_neurons_L2_vis[layer_idx].append(specific_neurons_L2_layer)

    """ flatten lists for visualization"""
    for layer_idx in range(num_layers):
        activated_neurons_L1_vis[layer_idx] = set(list(itertools.chain.from_iterable(activated_neurons_L1_vis[layer_idx])))
        activated_neurons_L2_vis[layer_idx] = set(list(itertools.chain.from_iterable(activated_neurons_L2_vis[layer_idx])))
        non_activated_neurons_L1_vis[layer_idx] = set(list(itertools.chain.from_iterable(non_activated_neurons_L1_vis[layer_idx])))
        non_activated_neurons_L2_vis[layer_idx] = set(list(itertools.chain.from_iterable(non_activated_neurons_L2_vis[layer_idx])))
        non_activated_neurons_all_vis[layer_idx] = set(list(itertools.chain.from_iterable(non_activated_neurons_all_vis[layer_idx])))
        shared_neurons_vis[layer_idx] = set(list(itertools.chain.from_iterable(shared_neurons_vis[layer_idx])))
        specific_neurons_L1_vis[layer_idx] = set(list(itertools.chain.from_iterable(specific_neurons_L1_vis[layer_idx])))
        specific_neurons_L2_vis[layer_idx] = set(list(itertools.chain.from_iterable(specific_neurons_L2_vis[layer_idx])))

    output_dict = {
        "activated_neurons_L1": activated_neurons_L1,
        "activated_neurons_L2": activated_neurons_L2,
        "non_activated_neurons_L1": non_activated_neurons_L1,
        "non_activated_neurons_L2": non_activated_neurons_L2,
        "shared_neurons": shared_neurons,
        "specific_neurons_L1": specific_neurons_L1,
        "specific_neurons_L2": specific_neurons_L2,
        "non_activated_neurons_all": non_activated_neurons_all
    }

    output_dict_vis = {
        "activated_neurons_L1": activated_neurons_L1_vis,
        "activated_neurons_L2": activated_neurons_L2_vis,
        "non_activated_neurons_L1": non_activated_neurons_L1_vis,
        "non_activated_neurons_L2": non_activated_neurons_L2_vis,
        "shared_neurons": shared_neurons_vis,
        "specific_neurons_L1": specific_neurons_L1_vis,
        "specific_neurons_L2": specific_neurons_L2_vis,
        "non_activated_neurons_all": non_activated_neurons_all_vis
    }

    return output_dict, output_dict_vis


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
