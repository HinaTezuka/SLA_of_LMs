import os
import random
import sys
import dill as pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import transformers
from baukit import TraceDict
from datasets import get_dataset_config_names, load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_out_llama3_act_fn(model, prompt, device, index):
    model.eval() # swith the model to evaluation mode (deactivate dropout, batch normalization)
    num_layers = model.config.num_hidden_layers  # nums of layers of the model
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(num_layers)]  # generate path to MLP layer(of LLaMA-3)

    torch.manual_seed(42)
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

    torch.manual_seed(42)
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
            output[:, -1, neuron_idx] *= 0  # 指定されたニューロンの活性化値をゼロに設定

    return output

def get_act_patterns_with_edit_activation(model, tokenizer, device, layer_neuron_list, data, model_type):
    # layers_num
    num_layers = 40 if model_type == 'phi4' else 32
    # nums of total neurons (per a layer)
    num_neurons = 17920 if model_type == 'phi4' else 14336
    act_patterns_dict = defaultdict(list)
    trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]))
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
        with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:
            act_fn_value_L2, up_proj_value_L2 = act_llama3(model, input_ids_L2)
        """
        neurons(in llama3 MLP): act_fn(gate_proj(x)) * up_proj(x) <- input to down_proj()
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
            """ get neuron activation values: act_fn(gate_proj(x)) * up_proj(x) """
            neurons_L1_values = calc_element_wise_product(act_fn_value_L1[layer_idx], up_proj_value_L1[layer_idx]) # torch.Tensor
            neurons_L2_values = calc_element_wise_product(act_fn_value_L2[layer_idx], up_proj_value_L2[layer_idx])
            """ calc act_patterns (cos_sim). """ 
            act_patterns = cosine_similarity(neurons_L1_values, neurons_L2_values)[0, 0]
            act_patterns_dict[layer_idx].append(act_patterns)

    return act_patterns_dict

def get_act_patterns(model, model_type, tokenizer, device, data):

    # layers_num
    num_layers = 40 if model_type == 'phi4' else 32
    # nums of total neurons (per a layer)
    num_neurons = 17920 if model_type == 'phi4' else 14336

    """
    activation valuesを保存する dict (shared neuronsが対象)
    {
        layer_idx: [cos_sim1, cos_sim2, ....] <- act_pattern(cos_sim of act_values): list
    }
    """
    act_patterns_dict = defaultdict(list)

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
        neurons(in llama3 MLP): act_fn(gate_proj(x)) * up_proj(x) <- input to down_proj()
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
            """ get neuron activation values: act_fn(gate_proj(x)) * up_proj(x) """
            neurons_L1_values = calc_element_wise_product(act_fn_value_L1[layer_idx], up_proj_value_L1[layer_idx]) # torch.Tensor
            neurons_L2_values = calc_element_wise_product(act_fn_value_L2[layer_idx], up_proj_value_L2[layer_idx])
            """ calc act_patterns (cos_sim). """ 
            act_patterns = cosine_similarity(neurons_L1_values, neurons_L2_values)[0, 0]
            act_patterns_dict[layer_idx].append(act_patterns)

    return act_patterns_dict

# def get_act_patterns_with_edit_activation(model, tokenizer, device, layer_neuron_list, data, model_type):
#     trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]))
#     with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:
#         return get_act_patterns(model, model_type,tokenizer, device, data)

# def get_act_patterns(model, model_type, tokenizer, device, data):

#     # layers_num
#     num_layers = 40 if model_type == 'phi4' else 32
#     # nums of total neurons (per a layer)
#     num_neurons = 17920 if model_type == 'phi4' else 14336

#     """
#     activation valuesを保存する dict (shared neuronsが対象)
#     {
#         layer_idx: [cos_sim1, cos_sim2, ....] <- act_pattern(cos_sim of act_values): list
#     }
#     """
#     act_patterns_dict = defaultdict(list)

#     # Track neurons with tatoeba
#     for L1_text, L2_text in data:
#         """
#         get activation values
#         mlp_activation_L1/L2: [torch.Tensor(batch_size, sequence_length, num_neurons) * num_layers]
#         """
#         # L1 text
#         input_ids_L1 = tokenizer(L1_text, return_tensors="pt").input_ids.to(device)
#         token_len_L1 = len(input_ids_L1[0])
#         act_fn_value_L1, up_proj_value_L1 = act_llama3(model, input_ids_L1)

#         # L2 text
#         input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to(device)
#         token_len_L2 = len(input_ids_L2[0])
#         act_fn_value_L2, up_proj_value_L2 = act_llama3(model, input_ids_L2)
#         """
#         neurons(in llama3 MLP): act_fn(gate_proj(x)) * up_proj(x) <- input to down_proj()
#         L1/L2 shared neurons
#         """
#         for layer_idx in range(len(act_fn_value_L1)):
#             """ consider last token only """
#             # L1
#             act_fn_value_L1[layer_idx] = act_fn_value_L1[layer_idx][:, token_len_L1 - 1, :]
#             up_proj_value_L1[layer_idx] = up_proj_value_L1[layer_idx][:, token_len_L1 - 1, :]
#             # L2
#             act_fn_value_L2[layer_idx] = act_fn_value_L2[layer_idx][:, token_len_L2 - 1, :]
#             up_proj_value_L2[layer_idx] = up_proj_value_L2[layer_idx][:, token_len_L2 - 1, :]
#             """ get neuron activation values: act_fn(gate_proj(x)) * up_proj(x) """
#             neurons_L1_values = calc_element_wise_product(act_fn_value_L1[layer_idx], up_proj_value_L1[layer_idx]) # torch.Tensor
#             neurons_L2_values = calc_element_wise_product(act_fn_value_L2[layer_idx], up_proj_value_L2[layer_idx])
#             """ calc act_patterns (cos_sim). """ 
#             act_patterns = cosine_similarity(neurons_L1_values, neurons_L2_values)[0, 0]
#             act_patterns_dict[layer_idx].append(act_patterns)

#     return act_patterns_dict

def get_act_patterns_inputs_to_down_proj(model, model_type, tokenizer, device, data):

    # layers_num
    num_layers = 40 if model_type == 'phi4' else 32
    # nums of total neurons (per a layer)
    num_neurons = 17920 if model_type == 'phi4' else 14336

    """
    activation valuesを保存する dict (shared neuronsが対象)
    {
        layer_idx: [cos_sim1, cos_sim2, ....] <- act_pattern(cos_sim of act_values): list
    }
    """
    act_patterns_dict = defaultdict(list)

    # Track neurons with tatoeba
    for L1_text, L2_text in data:
        # L1 text
        input_ids_L1 = tokenizer(L1_text, return_tensors="pt").input_ids.to(device)
        token_len_L1 = len(input_ids_L1[0])

        # L2 text
        input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to(device)
        token_len_L2 = len(input_ids_L2[0])

        """ get activation values: L1 """
        act_values_L1 = [] # len: layer_num
        # hook fn
        def get_inputs_to_down_proj_L1(model, input):
            act_values_L1.append(input[0][0][-1].detach().cpu().numpy())
        handles = []
        for layer_idx, layer in enumerate(model.model.layers):
            handle = layer.mlp.down_proj.register_forward_pre_hook(get_inputs_to_down_proj_L1)
            handles.append(handle)

        torch.manual_seed(42)
        # run inference.
        with torch.no_grad():
            L1_output = model(input_ids_L1)
        # remove hook.
        for handle in handles:
            handle.remove()
        """ get activation values: L2 """
        act_values_L2 = [] # len: layer_num
        # hook fn
        def get_inputs_to_down_proj_L2(model, input):
            act_values_L2.append(input[0][0][-1].detach().cpu().numpy())
        handles = []
        for layer_idx, layer in enumerate(model.model.layers):
            handle = layer.mlp.down_proj.register_forward_pre_hook(get_inputs_to_down_proj_L2)
            handles.append(handle)
        # run inference.
        with torch.no_grad():
            L2_output = model(input_ids_L2)
        # remove hook.
        for handle in handles:
            handle.remove()

        """ calc cos_sim """
        for layer_idx in range(num_layers):
            """ consider last token only """
            # L1
            neurons_L1_values = act_values_L1[layer_idx]
            print(act_values_L1[layer_idx])
            sys.exit()
            # L2
            neurons_L2_values = act_values_L2[layer_idx]
            """ calc act_patterns (cos_sim). """ 
            act_patterns = cosine_similarity(neurons_L1_values, neurons_L2_values)[0, 0]
            act_patterns_dict[layer_idx].append(act_patterns)

    return act_patterns_dict

# def get_act_patterns_inputs_to_down_proj_with_edit_activations(model, model_type, tokenizer, device, data):

#     # layers_num
#     num_layers = 40 if model_type == 'phi4' else 32
#     # nums of total neurons (per a layer)
#     num_neurons = 17920 if model_type == 'phi4' else 14336

#     """
#     activation valuesを保存する dict (shared neuronsが対象)
#     {
#         layer_idx: [cos_sim1, cos_sim2, ....] <- act_pattern(cos_sim of act_values): list
#     }
#     """
#     act_patterns_dict = defaultdict(list)

#     # Track neurons with tatoeba
#     for L1_text, L2_text in data:
#         # L1 text
#         input_ids_L1 = tokenizer(L1_text, return_tensors="pt").input_ids.to(device)
#         token_len_L1 = len(input_ids_L1[0])

#         # L2 text
#         input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to(device)
#         token_len_L2 = len(input_ids_L2[0])

#         """ get activation values: L1 """
#         act_values_L1 = [] # len: layer_num
#         # hook fn
#         def get_inputs_to_down_proj_L1(model, input, layer_idx: int):
#             if input[0].shape[1] == token_len_L1:
#                 for neuron_idx in neurons[layer_idx]:
#             act_values_L1.append(input[0][0][-1].detach().cpu().numpy())
#         handles = []
#         for layer_idx, layer in enumerate(model.model.layers):
#             handle = layer.mlp.down_proj.register_forward_pre_hook(get_inputs_to_down_proj_L1)
#             handles.append(handle)
#         # run inference.
#         with torch.no_grad():
#             L1_output = model(input_ids_L1)
#         # remove hook.
#         for handle in handles:
#             handle.remove()
#         """ get activation values: L2 """
#         act_values_L2 = [] # len: layer_num
#         # hook fn
#         def get_inputs_to_down_proj_L2(model, input):
#             act_values_L2.append(input[0][0][-1].detach().cpu().numpy())
#         handles = []
#         for layer_idx, layer in enumerate(model.model.layers):
#             handle = layer.mlp.down_proj.register_forward_pre_hook(get_inputs_to_down_proj_L2)
#             handles.append(handle)
#         # run inference.
#         with torch.no_grad():
#             L2_output = model(input_ids_L2)
#         # remove hook.
#         for handle in handles:
#             handle.remove()

#         """ calc cos_sim """
#         for layer_idx in range(num_layers):
#             """ consider last token only """
#             # L1
#             neurons_L1_values = act_values_L1[layer_idx].unsqueeze()
#             print(act_values_L1[layer_idx])
#             sys.exit()
#             # L2
#             neurons_L2_values = act_values_L2[layer_idx]
#             """ calc act_patterns (cos_sim). """ 
#             act_patterns = cosine_similarity(neurons_L1_values, neurons_L2_values)[0, 0]
#             act_patterns_dict[layer_idx].append(act_patterns)

#     return act_patterns_dict


def activation_patterns_lineplot(act_patterns, act_patterns_baseline, L2, intervention_num, model_type, score_type, intervention, is_reverse=False):
    """
    Plots line plots of cosine similarity means(Activation Patterns) per each layer for two dictionaries(main and baseline),
    with variance (standard deviation) as a shaded region.

    Parameters:
    - act_patterns: Dictionary in the format {layer_idx: [cos_sim1, cos_sim2, ...]}
    - act_patterns_baseline: Dictionary in the format {layer_idx: [cos_sim1, cos_sim2, ...]}
    """
    # Get all unique layer indices
    all_layer_idxs = [i for i in range(32)]

    def compute_stats(data_dict, layers):
        """ Compute mean and standard deviation for each layer index. """
        means, std_devs = [], []
        for layer in layers:
            values = data_dict.get(layer)
            means.append(np.mean(values))
            std_devs.append(np.std(values))
        return np.array(means), np.array(std_devs)

    # Compute mean and standard deviation for each dictionary
    means1, std_devs1 = compute_stats(act_patterns, all_layer_idxs)
    means2, std_devs2 = compute_stats(act_patterns_baseline, all_layer_idxs) # baseline

    # Plot
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.figure(figsize=(8, 7))
    
    # Plot act_patterns
    plt.plot(all_layer_idxs, means1, marker='o', linestyle='-', color='blue', label="Same Semantics")
    plt.fill_between(all_layer_idxs, means1 - std_devs1, means1 + std_devs1, color='blue', alpha=0.2)

    # Plot baseline
    plt.plot(all_layer_idxs, means2, marker='s', linestyle='--', color='red', label="Different Semantics")
    plt.fill_between(all_layer_idxs, means2 - std_devs2, means2 + std_devs2, color='red', alpha=0.2)

    # Labels and title
    plt.xlabel("Layer Index", fontsize=35)
    plt.ylabel("Cosine Sim", fontsize=35)
    plt.ylim(-0.1, 1)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.title(f"en_{L2}", fontsize=35)
    plt.legend(fontsize=20)
    plt.grid(True)
    if is_reverse:
        if intervention == "yes":
            save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/act_patterns/{model_type}/{score_type}/reverse/intervention/{L2}_n{intervention_num}"
        elif intervention == "baseline":
            save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/act_patterns/{model_type}/{score_type}/reverse/intervention_baseline/{L2}_n{intervention_num}"
    else:
        # save_path
        if intervention == "normal":
            save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/act_patterns/{model_type}/normal/{L2}"
        elif intervention == "yes":
            save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/act_patterns/{model_type}/{score_type}/intervention/{L2}_n{intervention_num}"
        elif intervention == "baseline":
            save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/act_patterns/{model_type}/{score_type}/intervention_baseline/{L2}_n{intervention_num}"
    
    with PdfPages(save_path + '.pdf') as pdf:
        pdf.savefig(bbox_inches='tight', pad_inches=0.01)
        plt.close()

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

def save_as_pickle(file_path, target_dict) -> None:
    """
    save dict as pickle file.
    """
    # directoryを作成（存在しない場合のみ)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(target_dict, f)

def unfreeze_pickle(file_path: str) -> dict:
    """
    unfreeze pickle file as dict.
    """
    with open(file_path, "rb") as f:
        return_dict = pickle.load(f)
    return return_dict
