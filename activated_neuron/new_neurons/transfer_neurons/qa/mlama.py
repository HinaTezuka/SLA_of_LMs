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
model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3']
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
qa = load_dataset('atutej/m_lama', 'ja')['test']
"""
available langs: 
['af', 'ar', 'az', 'be', 'bg', 'bn', 'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'ga', 'gl', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'it', 'ja', 'ka', 'ko', 'la', 'lt', 'lv', 'ms', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sq', 'sr', 'sv', 'ta', 'th', 'tr', 'transliterated-hi', 'uk', 'ur', 'vi', 'zh']
"""
# qa = qa.shuffle(seed=42)
def set_prompt_mlama(template: str, filler: str, XorY: str) -> str:
    if XorY == 'x':
        return template.replace('[X]', filler)
    elif XorY == 'y':
        return template.replace('[Y]', filler)
    
score_type = 'cos_sim'
intervention_num = 1000

XorY = 'x'
temps = {
    'ja': 'は',
    'ko': '는',
    'nl': 'is ',
    'it': 'è ',
    'en': 'is '
}
def mlama(model, tokenizer, device, qa, XorY: str, L2: str):
    for item in qa:
        filler = item['sub_label'] if XorY == 'x' else item['obj_label']
        XorY_reverse = '[Y]' if XorY == 'x' else '[X]'
        template = set_prompt_mlama(item['template'], filler, XorY)
        temp_is = temps[L2]
        # prompt = f'{template}\n{XorY_reverse}{temp_is}'
        prompt = f'{template}\npredict [Y].\n'
        # run inference.
        torch.cuda.manual_seed_all(42) # set seed.
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        pre = tokenizer.decode(output[0], skip_special_tokens=True)
        print(pre)
        ans = item['obj_label']
        print(f'ans: {ans}')
        # sys.exit()
results = {}
resutls_intervention = {}
resutls_intervention_baseline = {}
for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral'
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for L2 in langs:
        # normal
        result_score = mlama(model, tokenizer, device, qa, XorY, L2)
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