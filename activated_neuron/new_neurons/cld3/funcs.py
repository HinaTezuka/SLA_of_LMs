import os
import sys
import dill as pickle

import cld3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from baukit import TraceDict

def project_hidden_emb_to_vocab(model, tokenizer, hidden_states: torch.Tensor, last_token_index: int, top_k: int) -> dict:
    token_preds_dict = {} # {layer_idx: [token1, token2, ...]}
    for layer_idx in range(1, 33): # <- remove embedding layer(0th layer).
        hidden_state = hidden_states[layer_idx][0, last_token_index]
        # normalization
        normed = model.model.norm(hidden_state)
        # get logits
        logits = torch.matmul(model.lm_head.weight, normed.T)
        # make distribution
        probs = F.softmax(logits, dim=-1)
        probs = torch.reshape(probs, (-1,)).cpu().detach().numpy()

        probs_ = [] # [ (token_idx, probability), (), ..., () ]
        for index, prob in enumerate(probs):
            probs_.append((index, prob))

        top_k_tokens = sorted(probs_, key=lambda x: x[1], reverse=True)[:top_k]
        token_preds = [tokenizer.decode(t[0]) for t in top_k_tokens]

        token_preds_dict[layer_idx] = token_preds

    return token_preds_dict

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

def get_hidden_states_with_edit_activation(model, inputs, layer_neuron_list):
    trace_layers = [f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:
        # run inference
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
    return output.hidden_states

def print_tokens(token_dict):
    for k, v in token_dict.items():
        print(f"=========================== {k} ===========================")
        print(v, "\n")

def layerwise_lang_stats(token_dict, L2, L1="en"):
    lang_stats = {}
    for layer, tokens in token_dict.items():
        lang_stats[layer] = {'total_count': 0, L1: 0, L2: 0}
        for token in tokens:
            lang_pred = cld3.get_language(token)
            if lang_pred and lang_pred.is_reliable:
                lang_stats[layer]['total_count'] += 1
                if lang_pred.language == L1:
                    lang_stats[layer][L1] += 1
                elif lang_pred.language == L2:
                    lang_stats[layer][L2] += 1
    return lang_stats

def layerwise_lang_distribution(lang_stats, L2, L1="en"):
    lang_distribution = {}
    for layer, stats in lang_stats.items():
        if stats['total_count'] > 0:
            lang_distribution[layer] = {
                L1: stats[L1] / stats['total_count'],
                L2: stats[L2] / stats['total_count']
            }
        else:
            lang_distribution[layer] = {L1: 0, L2: 0}
    return lang_distribution

def plot_lang_distribution(lang_distribution, activation_type: str, intervention_type: str, intervention_num: int, L2: str, L1="en"):
    layers = sorted(lang_distribution.keys())
    en_values = [lang_distribution[layer][L1] for layer in layers]
    non_en_values = [lang_distribution[layer][L2] for layer in layers]
    
    lang_matrix = np.array([en_values, non_en_values])
    
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(lang_matrix, ax=ax, xticklabels=layers, yticklabels=[L1, L2], cmap='Blues', annot=False)
    plt.title('Layerwise Language Distribution')
    plt.xlabel('Layer Index')
    plt.ylabel('Language')
    plt.show()
    plt.savefig(
        f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/cld3/{activation_type}/{intervention_type}/{L2}_n{intervention_num}.png',
        bbox_inches='tight',
    )

def unfreeze_pickle(file_path: str) -> dict:
    """
    unfreeze pickle file as dict.
    """
    with open(file_path, "rb") as f:
        return_dict = pickle.load(f)
    return return_dict