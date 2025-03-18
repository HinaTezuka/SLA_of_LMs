"""
compare acc of QA for both normal and deactivated model.
"""
import os
import sys
import pickle
import collections

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

from qa_funcs import (
    compute_f1,
    mkqa,
    mkqa_with_edit_activation,
    save_as_pickle,
    unfreeze_pickle,
)

# load models (LLaMA3-8B).
model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
langs = ['ja', 'nl', 'ko', 'it']
""" 
QA dataset: 
MKQA: Multilingual Open Domain Question Answering
・https://arxiv.org/abs/2007.15207
・https://github.com/apple/ml-mkqa/
・https://huggingface.co/datasets/apple/mkqa
"""
qa_num = 1000
qa = load_dataset('apple/mkqa')['train']
score_type = 'cos_sim'
intervention_num = 1000

results = {}
resutls_intervention = {}
for model_name in model_names:
    if 'llama' in model_name: model_type = 'llama3'
    elif 'mistral' in model_name: model_type = 'mistral'
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for L2 in langs:
        # normal
        result_score = mkqa(model, tokenizer, device, qa, L2, qa_num)
        results[L2] = result_score
        # intervention
        intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
        intervened_neurons = unfreeze_pickle(intervened_neurons_path)[:intervention_num]
        result_score = mkqa_with_edit_activation(model, tokenizer, device, qa, L2, qa_num, intervened_neurons)
        resutls_intervention[L2] = result_score

    # save results as pkl.
    path_normal = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/all_langs.pkl'
    save_as_pickle(path_normal, results)
    path_intervention = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/all_langs_intervention.pkl'
    save_as_pickle(path_intervention, resutls_intervention)

    # print results (just in case).
    print(f'normal: {results}')
    print(f'intervention: {resutls_intervention}')

    del model
    torch.cuda.empty_cache()