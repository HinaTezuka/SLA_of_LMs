import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/hidden_states_classification")
import dill as pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    save_as_pickle,
    unfreeze_pickle,
)

# LLaMA-3(8B)
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}

for L2, model_name in model_names:
    """ logistic regression results """
    regression_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/logistic_regression/en_{L2}.pkl"
    regression_results = unfreeze_pickle(regression_path)

    """ svm results """
    svm_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/svm/en_{L2}.pkl"
    svm_results = unfreeze_pickle(svm_path)