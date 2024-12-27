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

"""
モデルの各層のhidden stateを取得・対訳ペアと非対訳ペアでそれぞれ類似度を測定
"""
import os
import sys
# sys.path.append("/home/s2410121/proj_LA/activated_neuron")
import dill as pickle

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def get_hidden_states(model, tokenizer, device, data, is_norm=False) -> list:
    """
    extract hidden states only for last tokens.
    is_norm: whether normalization of hidden states is required or not.

    return: np.array for input of sklearn classification models.
    """
    num_layers = 32
    # return
    input_for_sklearn_model = [[] for _ in range(num_layers)] # [layer_idx: 2000 pairs(translation or non translation)]

    for L1_txt, L2_txt in data:
        hidden_states = defaultdict(torch.Tensor)
        inputs_L1 = tokenizer(L1_txt, return_tensors="pt").to(device)
        inputs_L2 = tokenizer(L2_txt, return_tensors="pt").to(device)

        # get hidden_states
        with torch.no_grad():
            output_L1 = model(**inputs_L1, output_hidden_states=True)
            output_L2 = model(**inputs_L2, output_hidden_states=True)

        all_hidden_states_L1 = output_L1.hidden_states
        all_hidden_states_L2 = output_L2.hidden_states
        # get last token index.
        last_token_index_L1 = inputs_L1["input_ids"].shape[1] - 1
        last_token_index_L2 = inputs_L2["input_ids"].shape[1] - 1

        """ 各層の最後のトークンのhidden stateをリストに格納 + 正規化 """
        # last_token_hidden_states_L1 = np.array([
        #     (layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy() /
        #     np.linalg.norm(layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy(), axis=-1, keepdims=True))
        #     # embedding層(0層目)は排除
        #     for layer_hidden_state in all_hidden_states_L1[1:]
        # ])
        # last_token_hidden_states_L2 = np.array([
        #     (layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy() /
        #     np.linalg.norm(layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy(), axis=-1, keepdims=True))
        #     # embedding層(0層目)は排除
        #     for layer_hidden_state in all_hidden_states_L2[1:]
        # ])
        last_token_hidden_states_L1 = np.array([
            layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy()
            for layer_hidden_state in all_hidden_states_L1[1:]
            ])
        last_token_hidden_states_L2 = np.array([
            layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy()
            for layer_hidden_state in all_hidden_states_L2[1:]
            ])
        """ make features per a layer and save it to list. """
        for i in range(num_layers):
            # 1次元化
            feature_L1 = last_token_hidden_states_L1[i][0]
            feature_L2 = last_token_hidden_states_L2[i][0]
            # concatenate L1 and L2 features
            features_L1_and_L2 = np.concatenate([feature_L1, feature_L2]) # 4096 + 4096 -> 8192次元
            # features_L1_and_L2 = np.array([
            #     feature_L1, feature_L2
            # ])
            input_for_sklearn_model[i].append(features_L1_and_L2)

    return input_for_sklearn_model # shape: (num_layers, num_pairs, 8192)

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
