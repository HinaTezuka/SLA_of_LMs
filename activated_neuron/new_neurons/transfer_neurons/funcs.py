"""
neuron detection for MLP Block of LLaMA-3(8B).
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

def multilingual_dataset(langs: list, num_sentences=500) -> list:
    """
    making tatoeba translation corpus for lang-specific neuron detection.
    """
    L1 = "en"

    tatoeba_data = []
    for lang in langs:
        if lang == "en":
            dataset = load_dataset("tatoeba", lang1="en", lang2="ja", split="train")
            dataset = dataset.select(range(num_sentences)) 
            for item in dataset:
                if item['translation'][lang] != '':
                    tatoeba_data.append(item['translation'][lang])
            continue
        dataset = load_dataset("tatoeba", lang1="en", lang2=lang, split="train")
        dataset = dataset.select(range(num_sentences))
        for sentence_idx, item in enumerate(dataset):
            if item['translation'][lang] != '':
                tatoeba_data.append(item['translation'][lang])
    
    return tatoeba_data

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

def track_neurons_with_text_data(model, device, tokenizer, data):

    # layers_num
    num_layers = 32
    # nums of total neurons (per a layer)
    num_neurons = 14336
    # return dict for saving activation values.
    activation_dict = defaultdict(lambda: defaultdict(list))
    """
    activation_dict
    {
        text_idx:
            layer_idx: [(neuron_idx, act_value), (neuron_idx, act_value), ....]
    }
    """

    # Track neurons with tatoeba
    for text_idx, text in enumerate(data):
        """
        get activation values
        mlp_activation: [torch.Tensor(batch_size, sequence_length, num_neurons) * num_layers]
        """
        # input text
        inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)
        token_len = len(inputs[0])
        act_fn_values, up_proj_values = act_llama3(model, inputs)
        """
        neurons(in llama3 MLP): up_proj(x) * act_fn(gate_proj(x)) <- input of down_proj()
        """
        for layer_idx in range(len(act_fn_values)):
            """ consider average of all tokens """
            act_values_per_token = []
            for token_idx in range(token_len):
                act_fn_value = act_fn_values[layer_idx][:, token_idx, :][0]
                up_proj_value = up_proj_values[layer_idx][:, token_idx, :][0]

                """ calc and extract neurons: up_proj(x) * act_fn(x) """
                act_values = calc_element_wise_product(act_fn_value, up_proj_value)
                act_values_per_token.append(act_values)
            
            """ calc average act_values of all tokens """
            act_values_per_token = np.array(act_values_per_token)
            means = np.mean(act_values_per_token, axis=0)

            # save in activation_dict
            for neuron_idx in range(num_neurons):
                mean_act_value = means[neuron_idx]
                activation_dict[text_idx][layer_idx].append((neuron_idx, mean_act_value))

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

def compute_ap_and_sort(activations_dict, start_label1: int, num_sentences_per_L2: int):
    """
    calc APscore and sort (considering nums_of_label) for detecting L2-specific neurons.

    Args:
        activations: activation_value for each sentences.
        start_label1: indicating idx where label1(corresponding to L2) sentences bigin.

    Returns:
        sorted_neurons list, ニューロンごとのAPスコア辞書
    """
    # 各ニューロンごとの活性化値とラベルを収集
    neuron_responses = defaultdict(list)  # { (layer_idx, neuron_idx): [activation_values, ...] }
    neuron_labels = defaultdict(list)     # { (layer_idx, neuron_idx): [labels, ...] }
    end_label1 = start_label1 + num_sentences_per_L2

    # pairs for label:1
    for sentence_idx, layer_data in activations_dict.items():
        for layer_idx, neuron_activations in layer_data.items():
            for neuron_idx, activation_value in neuron_activations:
                neuron_responses[(layer_idx, neuron_idx)].append(activation_value)
                if sentence_idx >= start_label1 and sentence_idx < end_label1: # check if sentence_idx is in label1 sentences.
                    neuron_labels[(layer_idx, neuron_idx)].append(1)  # label1: L2 sentence.
                else:
                    neuron_labels[(layer_idx, neuron_idx)].append(0)  # label0: sentence in other langs.

    # calc AP score for each shared neuron and calc total score which consider nums of label
    final_scores = {}
    for (layer_idx, neuron_idx), activations in neuron_responses.items():
        labels = neuron_labels[(layer_idx, neuron_idx)]

        # calc AP score
        ap = average_precision_score(y_true=labels, y_score=activations)
        # save total score
        final_scores[(layer_idx, neuron_idx)] = ap

    # sort: based on total score of each neuron
    sorted_neurons = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)

    return sorted_neurons, final_scores
