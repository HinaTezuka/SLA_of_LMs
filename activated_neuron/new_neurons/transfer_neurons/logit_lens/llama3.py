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
  save_as_json,
  save_as_pickle,
  unfreeze_pickle,
)

""" model configs """
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
layer_nums = 32
score_types = ["L2_dis", "cos_sim"]
score_types = ["cos_sim"]
norm_type = "no"
top_n = 10
langs = ["ja", "nl", "ko", "it"]
# langs = ["ja"]
model_type = "llama3"

for L2 in langs:
    for score_type in score_types:
        pkl_file_path = f"activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/{score_type}/{L2}_revised.pkl"
        sorted_neurons_AP = unfreeze_pickle(pkl_file_path)[:top_n]
        # baseline
        # sorted_neurons_AP_baseline = random.sample(sorted_neurons_AP[top_n_for_baseline+1:], len(sorted_neurons_AP[top_n_for_baseline+1:]))

        """ get value_predictions """
        value_predictions = {}
        # for top_n AP neurons
        print(f"================ {L2}/{score_type}. ================")
        for neuron in sorted_neurons_AP[-top_n:]:
            layer_idx, neuron_idx = neuron[0], neuron[1]
            value_preds = project_value_to_vocab(model, tokenizer, layer_idx, neuron_idx, top_k=20, normed=True)
            value_predictions[(layer_idx, neuron_idx)] = value_preds
            print(f"================ {(layer_idx, neuron_idx)} ================")
            print(value_preds, "\n")
        sys.exit()

        # for baselines
        # print(f"================ baseline. ================")
        # value_predictions_baseline = {}
        # for neuron in sorted_neurons_AP_baseline[:top_n]:
        #   layer_idx, neuron_idx = neuron[0], neuron[1]
        #   value_preds_baseline = project_value_to_vocab(model, tokenizer, layer_idx, neuron_idx, normed=True)
        #   value_predictions_baseline[(layer_idx, neuron_idx)] = value_preds_baseline
        #   print(f"================ {(layer_idx, neuron_idx)} ================")
        #   print(value_preds_baseline, "\n")
        # sys.exit()

        """ delete model (for saving memory). """
        del model

        """ save as pickle. """
        # # value_predictions(AP).
        # save_dir = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/logit_lens/results/llama/{activation_type}/n_{top_n}/{L2}_value_predictions.pkl"
        # save_as_pickle(save_dir, value_predictions)
        # # value_predictions_baseline.
        # save_dir_baseline = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/logit_lens/results/llama/{activation_type}/n_{top_n}/{L2}_value_predictions_baseline.pkl"
        # save_as_pickle(save_dir_baseline, value_predictions_baseline)

        # unfreeze for confirmation.
        # print(f"====================== {L2} ======================\n\n")
        # unfreeze_pickle(save_dir)
        # print("====================== baseline ======================\n\n")
        # unfreeze_pickle(save_dir_baseline)

        # delete chache
        torch.cuda.empty_cache()