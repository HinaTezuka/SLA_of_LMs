"""
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(32000, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2SdpaAttention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=32000, bias=False)
)
"""

"""
investigate what kind of information in the value vectors corresponding to the top_n AP neurons.
"""
import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/logit_lens")
import dill as pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
  project_value_to_vocab,
  get_embedding_for_token,
  unfreeze_pickle,
  save_as_json,
  unfreeze_json,
)

""" model configs """
# GPT-2-small
model_names = {
    # "base": "gpt2",
    # "ja": "rinna/japanese-gpt2-small", # ja
    # "de": "ml6team/gpt2-small-german-finetune-oscar", # ger
    "nl": "GroNLP/gpt2-small-dutch", # du
    "it": "GroNLP/gpt2-small-italian", # ita
    "fr": "dbddv01/gpt2-french-small", # fre
    "ko": "skt/kogpt2-base-v2", # ko
    "es": "datificate/gpt2-small-spanish" # spa
}

""" params """
device = "cuda" if torch.cuda.is_available() else "cpu"
layer_nums = 12
activation_type = "abs"
# activation_type = "product"
norm_type = "no"
top_n = 500
# L2 = "ja"

for L2, model_name in model_names.items():
  model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  # get top AP neurons (layer_idx, neuron_idx)
  pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/sorted_neurons_{L2}.pkl"
  sorted_neurons_AP = unfreeze_pickle(pkl_file_path)
  # baseline
  sorted_neurons_AP_baseline = random.sample(sorted_neurons_AP[top_n+1:], len(sorted_neurons_AP[top_n+1:]))

  """ get value_predictions """
  value_predictions = {}
  # for top_n AP neurons
  for neuron in sorted_neurons_AP[:top_n]:
    layer_idx, neuron_idx = neuron[0], neuron[1]
    value_preds = project_value_to_vocab(model, tokenizer, layer_idx, neuron_idx, top_k=20, normed=True)
    value_predictions[(layer_idx, neuron_idx)] = value_preds
    # print(f"================ {(neuron[0], neuron[1])} ================")
    # print(value_preds, "\n")

  # for baselines
  value_predictions_baseline = {}
  for neuron in sorted_neurons_AP_baseline[:top_n]:
    layer_idx, neuron_idx = neuron[0], neuron[1]
    value_preds_baseline = project_value_to_vocab(model, tokenizer, layer_idx, neuron_idx, normed=True)
    value_predictions_baseline[(layer_idx, neuron_idx)] = value_preds_baseline
    # print(f"================ {(neuron[0], neuron[1])} ================")
    # print(value_preds_baseline, "\n")

  """ save as json. """
  # value_predictions(AP).
  save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/logit_lens/results/llama/{activation_type}/n_{top_n}"
  file_name_with_path = os.path.join(save_dir, f"{L2}_value_predictions.json")
  save_as_json(value_predictions, file_name_with_path)
  # value_predictions_baseline.
  file_name_with_path_baseline = os.path.join(save_dir, f"{L2}_value_predictions_baseline.json")
  save_as_json(value_predictions_baseline, file_name_with_path_baseline)

  # unfreeze for confirmation.
  unfreeze_json(file_name_with_path)
  unfreeze_json(file_name_with_path_baseline)