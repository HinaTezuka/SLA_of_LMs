import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from mknn_funcs import (
    compute_mutual_knn,
)
from funcs import (
    save_as_pickle,
    unfreeze_pickle,
    unfreeze_np_arrays,
)

model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
langs = ['ja', 'nl', 'ko', 'it']
L1 = 'en'

for model_name in model_names:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    knn_scores = {}
    for L2 in langs:
        path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_multi_train.pkl'
        sentences = unfreeze_pickle(path)
        res = compute_mutual_knn(model, tokenizer, device, sentences, L1, L2) # res: [knn_score_layer1, knn_score_layer2, ...]
        print(f'=================={model_type}, {L2}==================')
        print(res)
        knn_scores[L2] = res
    
    path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/knn/res_all_langs.pkl'
    save_as_pickle(path, knn_scores)

    # clear cache.
    del model
    torch.cuda.empty_cache()