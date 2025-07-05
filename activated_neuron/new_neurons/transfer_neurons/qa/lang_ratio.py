import os
import sys
import pickle
import collections
import random
import json

import numpy as np
import torch
import cld3
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

from qa_funcs import (
    mkqa_all,
    mkqa_all_with_edit_activation,
    save_as_pickle,
    unfreeze_pickle,
)


input_lang = 'ja'
langs = ['ja', 'nl', 'ko']
model_types = ['llama3', 'aya']

for model_type in model_types:
    for is_type1 in [True, False]:
        for L2 in langs:
            c = 0
            # path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/{"type1" if is_type1 else "type2"}_{input_lang}_{L2}.json' if model_type == 'llama3' else f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/{"type1" if is_type1 else "type2"}_{input_lang}_{L2}_100.json'
            path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/normal_ja.json'
            with open(path, 'r', encoding='utf-8') as f:
                s = json.load(f)
            for pre in s:
                pred_lang = cld3.get_language(pre)
                if pred_lang is not None and pred_lang.is_reliable and pred_lang.language == input_lang:
                    c += 1
            # print results.
            print(f'{model_type}, {"type1" if is_type1 else "type2"}, input: {input_lang} deact: {L2}')
            print(f'{c / len(s)}, {c}件')