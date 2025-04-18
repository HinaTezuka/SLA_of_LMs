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
    mkqa_all,
    mkqa_all_with_edit_activation,
    save_as_pickle,
    unfreeze_pickle,
)

# load models (LLaMA3-8B).
model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b']
# model_names = ['CohereForAI/aya-expanse-8b', 'mistralai/Mistral-7B-v0.3']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
langs = ['ja', 'nl', 'ko', 'it', 'en']
""" 
QA dataset: 
MKQA: Multilingual Open Domain Question Answering
・https://arxiv.org/abs/2007.15207
・https://github.com/apple/ml-mkqa/
・https://huggingface.co/datasets/apple/mkqa
"""
# qa_num = 100
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

    for L2 in langs:
        # normal
        result_scores = mkqa_all(model, tokenizer, device, qa, L2)
        results[L2] = result_scores
    # save results as pkl.
    path_normal = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/all_questions.pkl'
    save_as_pickle(path_normal, results)

    # for L2 in langs:
    #     """ intervention """
    #     if L2 == 'en':
    #         continue
    #     # intervention
    #     intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
    #     intervened_neurons = unfreeze_pickle(intervened_neurons_path)
    #     intervened_neurons_main = intervened_neurons[:intervention_num]
    #     result_score = mkqa_all_with_edit_activation(model, tokenizer, device, qa, L2, intervened_neurons_main)
    #     resutls_intervention[L2] = result_score
    #     # intervention baseline.
    #     random.seed(42)
    #     intervened_neurons_baseline = random.sample(intervened_neurons[intervention_num+1:], len(intervened_neurons[intervention_num+1:]))
    #     intervened_neurons_baseline = intervened_neurons_baseline[:intervention_num]
    #     result_score = mkqa_all_with_edit_activation(model, tokenizer, device, qa, L2, intervened_neurons_baseline)
    #     resutls_intervention_baseline[L2] = result_score

    # path_intervention = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/intervention_n{intervention_num}/all_questions_intervention.pkl'
    # save_as_pickle(path_intervention, resutls_intervention)
    # path_intervention_baseline = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/intervention_n{intervention_num}/all_questions_intervention_baseline.pkl'
    # save_as_pickle(path_intervention_baseline, resutls_intervention_baseline)

    # print results (just in case).
    # print(f'normal: {results}')
    # print(f'intervention: {resutls_intervention}')
    # print(f'intervention baseline: {resutls_intervention_baseline}')

    print(f'{model_type} completed.')
    del model
    torch.cuda.empty_cache()