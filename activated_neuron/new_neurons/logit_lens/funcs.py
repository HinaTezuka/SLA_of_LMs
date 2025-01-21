import os
import sys
import dill as pickle

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def project_value_to_vocab(model, tokenizer, layer_idx, value_idx, top_k=10, normed=False):
  value_vector = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[value_idx]
  # normalization if it's required.
  value_vector = model.model.norm(value_vector) if normed else value_vector
  # get logits.
  logits = torch.matmul(model.lm_head.weight, value_vector.T)
  # make distribution based on logits.
  probs = F.softmax(logits, dim=-1)
  probs = torch.reshape(probs, (-1,)).cpu().detach().numpy()

  probs_ = [] # [ (token_idx, probability), (), ..., () ]
  for index, prob in enumerate(probs):
      probs_.append((index, prob))

  top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:top_k]
  value_preds = [(tokenizer.decode(t[0]), t[0]) for t in top_k]

  return value_preds

# def project_value_to_vocab_2(model, tokenizer, top_k=10, AP_list) -> dict:

#     def get_all_profected_values(model, E):
#         layer_down_proj_vals = [
#             model[f"model.layers.{layer_idx}.mlp.down_proj"].T
#             for layer_idx in range(32)
#         ]
#         values = []
#         for layer_idx in range(32):
#             for dim in range(4096):
#                 values.append(layer_down_proj_vals[layer][dim].unsqueeze(0))
#         values = torch.cat(values)
#         logits = E.matmul(values.T).T.numpy()

#         return logits.detach().cpu().numpy()
    
#     projections = {}
#     for neuron in AP_list:
#         ids = np.argsort(-logits[])


def get_embedding_for_token(token_idx):
    # get embeddings from token_idx.
    embedding = model.model.embed_tokens.weight[token_idx]
    return embedding

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