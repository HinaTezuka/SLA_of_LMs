"""
detect language specific neurons.
"""
import sys
import dill as pickle

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    unfreeze_pickle,

)

# LLaMA-3
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}
device = "cuda" if torch.cuda.is_available() else "cpu"

for L2, model_name in model_names.items():
    """
    activations for L2.
    activation_dict
    {
        text_idx:
            layer_idx: [(neuron_idx, act_value), (neuron_idx, act_value), ....]
    }
    """
    # unfreeze activations.
    # file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/activations/{L2}.pkl"
    # activations = unfreeze_pickle(file_path)
    
    # calc AP scores.
    sorted_neurons, 
    
