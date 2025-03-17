"""
compare acc of QA for both normal and deactivated model.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import collections

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

from qa_funcs import (
    compute_f1,
    mkqa_for_steer_output_lang,
    mkqa_for_steer_output_lang_normal,
    # mkqa_with_edit_activation_for_steer_output_lang,
    remove_intersec,
    save_as_pickle,
    unfreeze_pickle,
    unfreeze_np_arrays,
)

""" 
QA dataset: 
MKQA: Multilingual Open Domain Question Answering
・https://arxiv.org/abs/2007.15207
・https://github.com/apple/ml-mkqa/
・https://huggingface.co/datasets/apple/mkqa
"""
# load models (LLaMA3-8B).
model_name = 'mistralai/Mistral-7B-v0.3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_type = 'mistral'
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load QA dataset.
qa_num = 10
qa = load_dataset('apple/mkqa')['train']
score_type = 'cos_sim'
langs = ['ja', 'nl', 'ko', 'it']
langs = ['nl']
intervention_num = 2000

results = {}
resutls_intervention = {}
pair_patterns = {
    'ja': [('ja', 'nl'), ('ja', 'ko'), ('ja', 'it')],
    'nl': [('nl', 'ja'), ('nl', 'ko'), ('nl', 'it')],
    'ko': [('ko', 'ja'), ('ko', 'nl'), ('ko', 'it')],
    'it': [('it', 'ja'), ('it', 'nl'), ('it', 'ko')],
}

def get_mean_act_value(neurons: list, model_type: str):
    save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/activations/{L2}_last_token.npz"
    act_values_arr = unfreeze_np_arrays(save_path_activations)
    act_values = []
    for layer_i, neuron_i in neurons:
        act_values.append(np.mean(act_values_arr[layer_i, neuron_i, :]))
    
    return np.mean(np.array(act_values))

for L2 in langs:
    # normal
    # result_score = mkqa_for_steer_output_lang_normal(model, tokenizer, device, qa, L2, qa_num)

    # intervention
    pair_pattern = pair_patterns[L2]
    for pair in pair_pattern:
        # 
        lang_deactivation, lang_activation = pair[0], pair[1]
        print('====================== lang pair ======================')
        print(f'lang_deactivation: {lang_deactivation}, lang_activation: {lang_activation}\n')

        # neurons for deactivation.
        intervened_neurons_path_deactivation = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{lang_deactivation}_sorted_neurons.pkl"
        intervened_neurons_deactivation = unfreeze_pickle(intervened_neurons_path_deactivation)
        intervened_neurons_deactivation = [neuron for neuron in intervened_neurons_deactivation if neuron[0] in [ _ for _ in range(20, 32)]][:intervention_num] # 21-32 layers
        # neurons for forced activation.
        intervened_neurons_path_activation = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{lang_activation}_sorted_neurons.pkl"
        intervened_neurons_activation = unfreeze_pickle(intervened_neurons_path_activation)
        intervened_neurons_activation = [neuron for neuron in intervened_neurons_activation if neuron[0] in [ _ for _ in range(20, 32)]][:intervention_num]
        # activation value set for forced_activation.
        act_value = get_mean_act_value(intervened_neurons_activation, model_type)
        # print(act_value)
        # sys.exit()
        # remove duplications from intervened_neurons_deactivation
        intervened_neurons_deactivation = remove_intersec(intervened_neurons_deactivation, intervened_neurons_activation)
        intervened_neurons_deactivation = [('de', layer, neuron) for layer, neuron in intervened_neurons_deactivation]
        intervened_neurons_activation = [('ac', layer, neuron) for layer, neuron in intervened_neurons_activation]
        # generate outputs.
        result_score = mkqa_for_steer_output_lang(model, tokenizer, device, qa, L2, qa_num, intervened_neurons_deactivation, intervened_neurons_activation)

# save results as pkl.
# path_normal = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/all_langs.pkl'
# save_as_pickle(path_normal)
# path_intervention = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/qa/all_langs_intervention.pkl'
# save_as_pickle(path_intervention)

# print results (just in case).
print(f'normal: {results}')
print(f'intervention: {resutls_intervention}')

del model
torch.cuda.empty_cache()