import os
import sys
import pickle
import collections
import random
import json

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

from qa_funcs import (
    mkqa_entropy,
    mkqa_entropy_with_deactivation,
    save_as_pickle,
    unfreeze_pickle,
)

# load models.
# model_names = ['mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b']
model_names = ['CohereForAI/aya-expanse-8b', 'meta-llama/Meta-Llama-3-8B']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# langs = ['ja', 'nl', 'ko', 'it', 'en', 'vi']
langs = ['ja', 'nl', 'ko']
""" 
QA dataset: 
MKQA: Multilingual Open Domain Question Answering
・https://arxiv.org/abs/2007.15207
・https://github.com/apple/ml-mkqa/
・https://huggingface.co/datasets/apple/mkqa
"""

qa = load_dataset('apple/mkqa')['train']
# qa = qa.shuffle(seed=42)
score_type = 'cos_sim'
intervention_num = 1000
q_num = 20

results = {}
resutls_intervention = {}
resutls_intervention_baseline = {}
for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_lang = 'ja'
    metric = 'perplexity'
    # metric = 'entropy'

    # for L2 in langs:
    # normal
    result_scores = mkqa_entropy(model, tokenizer, device, qa, 'ja', metric)
    # save results as json.
    path_normal = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/confidence_scores/{metric}/{input_lang}_normal.json'
    with open(path_normal, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for L2 in langs:
        """ intervention """
        if L2 == 'en':
            continue
        # intervention
        intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
        intervened_neurons = unfreeze_pickle(intervened_neurons_path)
        intervened_neurons_main = intervened_neurons[:intervention_num]
        result_score_intervention = mkqa_entropy_with_deactivation(model, model_type, tokenizer, device, qa, input_lang, intervened_neurons_main, metric)
        
        # save as json.
        path_intervention = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/confidence_scores/{metric}/input_{input_lang}_deact_{L2}.json'
        path_normal = f'/home/ach17600st/SLA_of_LMs/activated_neuron/new_neurons/transfer_neurons/qa/outputs/{model_type}/normal_{input_lang}_{q_num}.json'
        with open(path_normal, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'{model_type} completed.')
    del model
    torch.cuda.empty_cache()