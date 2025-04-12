"""
some funcs were copied from: https://github.com/minyoungg/platonic-rep/blob/main/metrics.py
The Platonic Representation Hypothesis: https://arxiv.org/abs/2405.07987
"""

import sys

from datasets import load_dataset
import numpy as np
import torch
import torch.nn.functional as F

# Extract features from the models (example for embedding layer or first hidden layer)
def extract_representations(model, tokenizer, text, layer_idx=-1) -> torch.Tensor: # layer_indexで「どの層」のrepresentationsをとるかを指定
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states[1:]
        representations = hidden_states[layer_idx]
    # return representations.mean(dim=1)  # Mean-pooling across tokens
    return F.normalize(representations.mean(dim=1))

def mutual_knn(feats_A, feats_B, topk=5):
    """
    Computes the mutual KNN accuracy.

    Args:
        feats_A: A torch tensor of shape N x feat_dim
        feats_B: A torch tensor of shape N x feat_dim

    Returns:
        A float representing the mutual KNN accuracy
    """
    knn_A = compute_nearest_neighbors(feats_A, topk)
    knn_B = compute_nearest_neighbors(feats_B, topk)

    n = knn_A.shape[0]
    topk = knn_A.shape[1]

    # Create a range tensor for indexing
    range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

    # Create binary masks for knn_A and knn_B
    lvm_mask = torch.zeros(n, n, device=knn_A.device)
    llm_mask = torch.zeros(n, n, device=knn_A.device)

    lvm_mask[range_tensor, knn_A] = 1.0
    llm_mask[range_tensor, knn_B] = 1.0

    acc = (lvm_mask * llm_mask).sum(dim=1) / topk

    return acc.mean().item()

def compute_nearest_neighbors(feats, topk=5):
    """
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn

""" """
# def compute_mutual_knn_acc(
#     model,
#     tokenizer,
#     device,
#     texts_L1,
#     texts_L2,
#     topk
#     ):
#     # get representations.
#     feats_base = extract_representations(model, tokenizer, texts_L1)
#     feats_L2 = extract_representations(model, tokenizer, texts_L2)

#     return mutual_knn(feats_base, feats_L2, topk)

def knn(model, tokenizer, device, sentences: list, L1: str, L2: str) -> list:
    # layer_num = model.layer_num
    layer_num = 32
    sentences_num = len(sentences)
    hidden_dim_size = 4096 # dim_size of hidden states.
    feats_L1 = torch.zeros((layer_num, sentences_num, hidden_dim_size), device=device)
    feats_L2 = torch.zeros((layer_num, sentences_num, hidden_dim_size), device=device)
    for txt_idx, (L1_txt, L2_txt) in enumerate(sentences):
        inputs_L1 = tokenizer(L1_txt, return_tensors='pt').to(device)
        inputs_L2 = tokenizer(L2_txt, return_tensors='pt').to(device)
        # run inference.
        with torch.no_grad():
            output_L1 = model(**inputs_L1, output_hidden_states=True)
            output_L2 = model(**inputs_L2, output_hidden_states=True)
        hs_L1_all_layers = output_L1.hidden_states[1:]
        hs_L2_all_layers = output_L2.hidden_states[1:]
        for layer_idx in range(layer_num):
            hs_L1 = hs_L1_all_layers[layer_idx][:, -1, :].squeeze()
            hs_L2 = hs_L2_all_layers[layer_idx][:, -1, :].squeeze()
            feats_L1[layer_idx, txt_idx, :] = hs_L1
            feats_L2[layer_idx, txt_idx, :] = hs_L2
        
    # calc Mutual KNN.
    knn_scores = [] # [knn_score_layer1, knn_score_layer2, ...]
    for layer_idx in range(layer_num):
        feats_L1 = F.normalize(feats_L1, dim=-1) # 原論文はnormalizeしていたので.
        feats_L2 = F.normalize(feats_L2, dim=-1)
        knn_score = mutual_knn(feats_L1[layer_idx, :, :], feats_L2[layer_idx, :, :])
        knn_scores.append(knn_score)

    return knn_scores