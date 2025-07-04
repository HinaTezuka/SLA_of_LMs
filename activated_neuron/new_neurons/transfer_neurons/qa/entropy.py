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
from baukit import TraceDict

from qa_funcs import (
    mkqa_entropy,
    mkqa_entropy_with_deactivation,
    save_as_pickle,
    unfreeze_pickle,
)

# load models.
model_names = ['mistralai/Mistral-7B-v0.3']
model_names = ['CohereForAI/aya-expanse-8b', 'meta-llama/Meta-Llama-3-8B']
model_names = ['meta-llama/Meta-Llama-3-8B']
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
q_num = 50

for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    input_lang = 'ja'
    # metric = 'perplexity'
    metric = 'entropy'


    def edit_activation(output, layer, layer_idx_and_neuron_idx):
        """
        edit activation value of neurons(indexed layer_idx and neuron_idx)
        output: activation values
        layer: sth like 'model.layers.{layer_idx}.mlp.act_fn'
        layer_idx_and_neuron_idx: list of tuples like [(layer_idx, neuron_idx), ....]
        """
        for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
            if str(layer_idx) in layer and output.shape[1] != 1:
                output[:, -1, neuron_idx] *= 0

        return output
    
    intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/ja_mono_train.pkl"
    layer_neuron_list = unfreeze_pickle(intervened_neurons_path)[:1000]

    # for L2 in langs:
    # normal
    for i in range(len(qa['queries'])):
        q = qa['queries'][i][input_lang] # question
        if qa['answers'][i][input_lang][0]['aliases'] == []:
            a = [qa['answers'][i][input_lang][0]['text']]
        else:
            a = qa['answers'][i][input_lang][0]['aliases']
        def contains_none_or_empty(lst: list) -> bool:
            return any(x is None or x == '' for x in lst)

        if q == '' or q == None or contains_none_or_empty(a):
            continue

        # make prompt.
        if input_lang == 'ja': prompt = f'{q}? 答え: '
        elif input_lang == 'nl': prompt = f'{q}? Antwoord: '
        elif input_lang == 'ko': prompt = f'{q}? 답변: '
        elif input_lang == 'it': prompt = f'{q}? Risposta: '
        elif input_lang == 'en': prompt = f'{q}? Answer: '

        # run inference.
        torch.cuda.manual_seed_all(42)
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        print(inputs)
        trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]))
        with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:
            with torch.no_grad():
                output = model.generate(
                    **inputs, 
                    pad_token_id=tokenizer.eos_token_id,
                    )
        pre = tokenizer.decode(output[0], skip_special_tokens=True)
        # 
        if input_lang == 'ja': pre = pre.split("答え: ")[-1].strip()
        if input_lang == 'nl': pre = pre.split('Antwoord: ')[-1].strip()
        if input_lang == 'ko': pre = pre.split('답변: ')[-1].strip()
        if input_lang == 'it': pre = pre.split('Risposta: ')[-1].strip()
        if input_lang == 'en': pre = pre.split('Answer: ')[-1].strip()

        print(model_type)
        print(pre)

    result_scores = mkqa_entropy(model, tokenizer, device, qa, 'ja', metric, q_num)
    # save results as json.
    path_normal = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/confidence_scores/{metric}/{input_lang}_normal.json'
    with open(path_normal, "w", encoding="utf-8") as f:
        json.dump(result_scores, f, ensure_ascii=False, indent=2)

    for L2 in langs:
        """ intervention """
        if L2 == 'en':
            continue
        # intervention

        # intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
        intervened_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
        intervened_neurons = unfreeze_pickle(intervened_neurons_path)
        intervened_neurons = [neuron for neuron in intervened_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
        intervened_neurons_main = intervened_neurons[:intervention_num]
        result_score_intervention = mkqa_entropy_with_deactivation(model, model_type, tokenizer, device, qa, input_lang, intervened_neurons_main, metric, q_num)
        
        # save as json.
        # 分布全体のエントロピー
        path_intervention = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/confidence_scores/{metric}/input_{input_lang}_deact_{L2}_distribution.json'
        # generatedされたトークン1つのエントロピー
        # path_intervention = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/confidence_scores/{metric}/input_{input_lang}_deact_{L2}.json'
        with open(path_intervention, "w", encoding="utf-8") as f:
            json.dump(result_score_intervention, f, ensure_ascii=False, indent=2)

    print(f'{model_type} completed.')
    del model
    torch.cuda.empty_cache()