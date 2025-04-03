"""
compare acc of QA for both normal and deactivated model.
"""
import os
import sys
import pickle
import collections
import random

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

from qa_funcs import (
    get_f1_above_th_questions,
    calculate_f1,
    mkqa,
    mkqa_with_edit_activation,
    save_as_pickle,
    unfreeze_pickle,
)

# load models (LLaMA3-8B).
model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
langs = ['ja', 'nl', 'ko', 'it', 'en']
""" 
QA dataset: 
MKQA: Multilingual Open Domain Question Answering
・https://arxiv.org/abs/2007.15207
・https://github.com/apple/ml-mkqa/
・https://huggingface.co/datasets/apple/mkqa
"""
qa_num = 100
qa = load_dataset('apple/mkqa')['train']
# qa = qa.shuffle(seed=42)
score_type = 'cos_sim'
intervention_num = 1000

results = {}
resutls_intervention = {}
resutls_intervention_baseline = {}
for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    """ get question_indices whose f1_score exceeds THRESHOLD. """
    THRESHOLD = 0.5
    qa_dict = get_f1_above_th_questions(model, tokenizer, device, qa, langs, qa_num, THRESHOLD)
    save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qa/{model_type}_qa_indices_above_{THRESHOLD}_all_langs.pkl'
    save_as_pickle(save_path, qa_dict)
    print(f'successfully saved qa_Dict: qa_num{qa_num}, threshold{THRESHOLD}.')

    for L2 in langs:
        # normal
        result_score = mkqa(model, tokenizer, device, qa, L2, qa_num, qa_dict)
        results[L2] = result_score
        # intervention
        intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
        intervened_neurons = unfreeze_pickle(intervened_neurons_path)
        intervened_neurons_main = intervened_neurons[:intervention_num]
        result_score = mkqa_with_edit_activation(model, tokenizer, device, qa, L2, qa_num, intervened_neurons_main, qa_dict)
        resutls_intervention[L2] = result_score
        # intervention baseline.
        random.seed(42)
        intervened_neurons_baseline = random.sample(intervened_neurons[intervention_num+1:], len(intervened_neurons[intervention_num+1:]))
        intervened_neurons_baseline = intervened_neurons_baseline[:intervention_num]
        result_score = mkqa_with_edit_activation(model, tokenizer, device, qa, L2, qa_num, intervened_neurons_baseline, qa_dict)
        resutls_intervention_baseline[L2] = result_score

    # save results as pkl.
    path_normal = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/intervention_n{intervention_num}/all_langs_n{qa_num}_above{THRESHOLD}.pkl'
    save_as_pickle(path_normal, results)
    path_intervention = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/intervention_n{intervention_num}/all_langs_intervention_n{qa_num}_above{THRESHOLD}.pkl'
    save_as_pickle(path_intervention, resutls_intervention)
    path_intervention_baseline = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/intervention_n{intervention_num}/all_langs_intervention_baseline_n{qa_num}_above{THRESHOLD}.pkl'
    save_as_pickle(path_intervention_baseline, resutls_intervention_baseline)

    # print results (just in case).
    print(f'normal: {results}')
    print(f'intervention: {resutls_intervention}')
    print(f'intervention baseline: {resutls_intervention_baseline}')

    del model
    torch.cuda.empty_cache()