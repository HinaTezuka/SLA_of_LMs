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

def track_neurons_with_text_data(model, device, model_name, tokenizer, data, is_translation_pairs: bool, active_THRESHOLD=0.01):

    # layers_num
    num_layers = 32 if model_name == "llama" else 12
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
        neurons(in llama3 MLP): up_proj(x) * act_fn(x)
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
            neurons_L1 = torch.nonzero(torch.abs(neurons_L1_values) > active_THRESHOLD).cpu().numpy()
            neurons_L2 = torch.nonzero(torch.abs(neurons_L2_values) > active_THRESHOLD).cpu().numpy()

            """ shared neurons """
            shared_neurons_indices = np.intersect1d(neurons_L1[:, 1], neurons_L2[:, 1])

            """ get activation_value of each shared_neurons """
            for neuron_idx in shared_neurons_indices:
                act_value_L1 = get_activation_value(neurons_L1_values, neuron_idx)
                act_value_L2 = get_activation_value(neurons_L2_values, neuron_idx)
                activation_value = (act_value_L1 + act_value_L2) / 2
                # activation_dictに追加
                activation_dict[pair_idx][layer_idx].append((neuron_idx, activation_value))

        pair_idx += 1

    return activation_dict

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

def unfreeze_pickle(file_path: str) -> dict:
    """
    unfreeze pickle file as dict.
    """
    with open(file_path, "rb") as f:
        return_dict = pickle.load(f)
    return return_dict
