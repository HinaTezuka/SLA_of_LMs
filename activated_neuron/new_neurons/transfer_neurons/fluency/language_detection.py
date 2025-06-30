import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import defaultdict

import cld3

from funcs import (
    save_as_pickle,
    unfreeze_pickle,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

def lang_detection(sentence: str) -> bool:
    return cld3.get_language(sentence)


model_types = ['aya', 'llama3', 'mistral']
langs = ['ja', 'ko', 'fr']
is_baselines = [True, False]
num_to_extract = 100

for model_type in model_types:
    for L2 in langs:
        for is_baseline in is_baselines:
            save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{L2}.pkl' if not is_baseline else f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{L2}_baseline.pkl'
            sentences = unfreeze_pickle(save_path)
            cnt = 0
            for sentence in sentences:
                pred_language = lang_detection(sentence)
                if pred_language is not None and pred_language.is_reliable and pred_language.language == L2:
                    cnt += 1
            
            """ print results. """
            print(f'------------ {model_type}, {L2}, {"Baseline" if is_baseline else "Type-1"} ------------')
            print(f'proportion: {cnt / num_to_extract}, {cnt}/{num_to_extract}')