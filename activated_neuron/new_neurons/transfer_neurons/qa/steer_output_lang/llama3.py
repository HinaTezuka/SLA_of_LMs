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
    get_mean_act_value,
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
model_name = 'meta-llama/Meta-Llama-3-8B'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_type = 'llama3'
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load QA dataset.
qa_num = 10
qa = load_dataset('apple/mkqa')['train']
# qa = load_dataset('atutej/m_lama')
score_type = 'cos_sim'
langs = ['ja', 'nl', 'ko', 'it']
langs = ['it']
intervention_num = 1000
is_act = False

results = {}
resutls_intervention = {}
pair_patterns = {
    'ja': [('ja', 'nl'), ('ja', 'ko'), ('ja', 'it')],
    'nl': [('nl', 'ja'), ('nl', 'ko'), ('nl', 'it')],
    'ko': [('ko', 'ja'), ('ko', 'nl'), ('ko', 'it')],
    'it': [('it', 'ja'), ('it', 'nl'), ('it', 'ko')],
}

for L2 in langs:
    # normal
    # result_score = mkqa_for_steer_output_lang_normal(model, tokenizer, device, qa, L2, qa_num)

    # intervention
    pair_pattern = pair_patterns[L2]
    for pair in pair_pattern:
        # 
        lang_deactivation, lang_activation = pair[0], pair[1]
        print('====================== lang pair ======================')
        print(f'lang_deactivation: {lang_deactivation}, lang_activation: {lang_activation}')

        # neurons for deactivation.
        neurons_path_deactivation = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/reverse/{score_type}/{lang_deactivation}_sorted_neurons.pkl"
        neurons_deactivation = unfreeze_pickle(neurons_path_deactivation)
        neurons_deactivation = [neuron for neuron in neurons_deactivation if neuron[0] in [ _ for _ in range(25, 32)]][:intervention_num] # 21-32 layers
        # neurons for forced activation.
        neurons_path_activation = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/reverse/{score_type}/{lang_activation}_sorted_neurons.pkl"
        neurons_activation = unfreeze_pickle(neurons_path_activation)
        neurons_activation = [neuron for neuron in neurons_activation if neuron[0] in [ _ for _ in range(25, 32)]][:intervention_num]

        # activation value set for forced_activation.
        # act_values = get_mean_act_value(neurons_activation, lang_activation, model_type)

        # remove duplications from neurons_deactivation
        neurons_deactivation_removed = remove_intersec(neurons_deactivation, neurons_activation)
        # neurons_activation_removed = remove_intersec(neurons_activation, neurons_deactivation)

        neurons_deactivation_removed = [('de', layer, neuron) for layer, neuron in neurons_deactivation_removed]
        # neurons_activation_removed = [('ac', layer, neuron) for layer, neuron in neurons_activation_removed]
        # neurons_deactivation = [('de', layer, neuron) for layer, neuron in neurons_deactivation]
        neurons_activation = [('ac', layer, neuron) for layer, neuron in neurons_activation]
        # generate outputs.
        result_score = mkqa_for_steer_output_lang(model, tokenizer, device, qa, lang_deactivation, qa_num, neurons_deactivation_removed, neurons_activation)

del model
torch.cuda.empty_cache()