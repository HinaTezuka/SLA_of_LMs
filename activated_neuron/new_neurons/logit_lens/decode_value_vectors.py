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
# LLaMA-3(8B)
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}

""" params """
device = "cuda" if torch.cuda.is_available() else "cpu"
layer_nums = 32
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