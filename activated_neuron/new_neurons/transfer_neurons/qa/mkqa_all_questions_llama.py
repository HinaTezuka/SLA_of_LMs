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
model_names = ['meta-llama/Meta-Llama-3-8B']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
langs = ['ja', 'nl', 'ko', 'it']
# langs = ['it', 'ko']
# langs = ['ko']
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

    # for L2 in langs:
    #     # normal
    #     result_scores = mkqa_all(model, tokenizer, device, qa, L2)
    #     # save results as pkl.
    #     path_normal = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/all_questions_normal_{L2}.pkl'
    #     save_as_pickle(path_normal, result_scores)
    #     print(f'saved: normal: {model_type}, {L2}')

    for L2 in langs:
        """ intervention """
        if L2 == 'en':
            continue
        # intervention
        intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
        intervened_neurons = unfreeze_pickle(intervened_neurons_path)
        intervened_neurons_main = intervened_neurons[:intervention_num]
        # if L2 != 'ko':
        #     result_score_intervention = mkqa_all_with_edit_activation(model, tokenizer, device, qa, L2, intervened_neurons_main)
        #     path_intervention = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/intervention_n{intervention_num}/all_questions_intervention_{L2}.pkl'
        #     save_as_pickle(path_intervention, result_score_intervention)
        #     print(f'saved: intervention: {model_type}, {L2}')

        # intervention baseline.
        random.seed(42)
        intervened_neurons_baseline = random.sample(intervened_neurons[intervention_num:], intervention_num))
        intervened_neurons_baseline = intervened_neurons_baseline[:intervention_num]
        result_score_baseline = mkqa_all_with_edit_activation(model, tokenizer, device, qa, L2, intervened_neurons_baseline)
        path_intervention_baseline = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/intervention_n{intervention_num}/all_questions_baseline_{L2}.pkl'
        save_as_pickle(path_intervention_baseline, result_score_baseline)
        print(f'saved: baseline: {model_type}, {L2}')

    print(f'{model_type} completed.')
    del model
    torch.cuda.empty_cache()