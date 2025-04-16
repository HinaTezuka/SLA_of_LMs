"""
compare acc of QA for both normal and deactivated model.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import collections
import random

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

from qa_funcs import (
    get_mean_act_value,
    mkqa_for_steer_output_lang,
    mkqa_for_steer_output_lang_normal,
    mkqa_for_steer_output_lang_act_values,
    mkqa_for_steer_output_lang_add_subducted_vectors,
    mkqa_for_steer_output_lang_patching_with_elem_wise_product,
    mkqa_for_steer_output_lang_patching_with_elem_wise_product_tran_mean,
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
model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b']
model_names = ['CohereForAI/aya-expanse-8b']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load QA dataset.
qa_num = 100
qa = load_dataset('apple/mkqa')['train']
score_type = 'cos_sim'
# score_type = 'L2_dis'
langs = ['ja', 'nl', 'ko', 'it', 'en']
# langs = ['nl']
intervention_num = 1000

results = {} # normal(without intervention.)
resutls_intervention = {} # intervened ver.
pair_patterns = {
    'en': [('en', 'ja'), ('en', 'nl'), ('en', 'ko'), ('en', 'it')],
    'ja': [('ja', 'nl'), ('ja', 'ko'), ('ja', 'it')],
    'nl': [('nl', 'ja'), ('nl', 'ko'), ('nl', 'it')],
    'ko': [('ko', 'ja'), ('ko', 'nl'), ('ko', 'it')],
    'it': [('it', 'ja'), ('it', 'nl'), ('it', 'ko')],
}

lang_ratios_final = {}

for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for L2 in langs:
        # normal
        # results[L2], lang_ratios = mkqa_for_steer_output_lang_normal(model, tokenizer, device, qa, L2, qa_num)

        # intervention
        pair_pattern = pair_patterns[L2]
        for pair in pair_pattern:
            # 
            lang_deactivation, lang_activation = pair[0], pair[1]
            print('====================== lang pair ======================')
            print(f'lang_deactivation: {lang_deactivation}, lang_activation: {lang_activation}')

            # neurons for deactivation.
            neurons_path_deactivation = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{lang_deactivation}_sorted_neurons.pkl"
            neurons_deactivation = unfreeze_pickle(neurons_path_deactivation)
            neurons_deactivation = [neuron for neuron in neurons_deactivation if neuron[0] in [ _ for _ in range(20, 32)]][:intervention_num] # 21-32 layers
            # neurons for forced activation.
            neurons_path_activation = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{lang_activation}_sorted_neurons.pkl"
            neurons_activation = unfreeze_pickle(neurons_path_activation)
            neurons_activation = [neuron for neuron in neurons_activation if neuron[0] in [ _ for _ in range(20, 32)]][:intervention_num]

            # activation value set for forced_activation.
            # all_neurons = list(set(neurons_activation+neurons_deactivation))
            act_values_act = get_mean_act_value(lang_activation, model_type)

            # remove duplications from neurons_deactivation
            # neurons_deactivation_removed = remove_intersec(neurons_deactivation, neurons_activation)
            # neurons_activation_removed = remove_intersec(neurons_activation, neurons_deactivation)

            # neurons_deactivation_removed = [('de', layer, neuron) for layer, neuron in neurons_deactivation_removed]
            # neurons_activation_removed = [('ac', layer, neuron) for layer, neuron in neurons_activation_removed]
            neurons_deactivation = [('de', layer, neuron) for layer, neuron in neurons_deactivation]
            neurons_activation = [('ac', layer, neuron) for layer, neuron in neurons_activation]
            # generate outputs.
            # result_score = mkqa_for_steer_output_lang(model, tokenizer, device, qa, lang_deactivation, qa_num, neurons_deactivation_removed, neurons_activation)
            """ use act_value for translation Question. """
            # result_score = mkqa_for_steer_output_lang_act_values(model, tokenizer, device, qa, lang_deactivation, lang_activation, qa_num, neurons_deactivation, neurons_activation)
            """ add (c^l_lang2 - c^l_lang1) to hs of certain layer. """
            # load c_lang2(lang_deact) and c_lang1(lang_act)
            # c_langs = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train.pkl")
            # c_lang_activation = c_langs[lang_activation]
            # c_lang_deactivation = c_langs[lang_deactivation]

            # including en_centroids.
            # c_lang_activation = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_{lang_activation}.pkl")
            # c_lang_deactivation = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_{lang_deactivation}.pkl")

            # c_lang_deactivation = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_{lang_deactivation}_qa.pkl")
            # c_lang_activation = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_{lang_activation}_qa.pkl")

            # meaned subtrancted vector.
            sub_vectors = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/qa/c_qa_tran_{lang_deactivation}_{lang_activation}.pkl")
            
            # generate outputs.
            # resutls_intervention[(lang_deactivation, lang_activation)] = mkqa_for_steer_output_lang_add_subducted_vectors(model, tokenizer, device, qa, lang_deactivation, lang_activation, qa_num, neurons_deactivation, neurons_activation, c_lang_deactivation, c_lang_activation, act_values_act)
            # resutls_intervention[(lang_deactivation, lang_activation)], lang_ratios = mkqa_for_steer_output_lang_patching_with_elem_wise_product(model, tokenizer, device, qa, lang_deactivation, lang_activation, qa_num, neurons_deactivation, neurons_activation, c_lang_deactivation, c_lang_activation, act_values_act=act_values_act)
            # lang_ratios_final[(lang_deactivation, lang_activation)] = lang_ratios

            # 対訳同士の差ベクトルの平均を足す
            resutls_intervention[(lang_deactivation, lang_activation)], lang_ratios = mkqa_for_steer_output_lang_patching_with_elem_wise_product_tran_mean(model, tokenizer, device, qa, lang_deactivation, lang_activation, qa_num, neurons_deactivation, neurons_activation, sub_vectors, act_values_act=act_values_act)
            lang_ratios_final[(lang_deactivation, lang_activation)] = lang_ratios

    # save_path_normal = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qa/{model_type}/lang_ratio/normal_n{intervention_num}_19_mean_patching.pkl'
    save_path_intervention = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qa/{model_type}/lang_ratio/intervention_n{intervention_num}_add_subtracted_vectors_to_last_two_layers_only.pkl'
    # save_as_pickle(save_path_normal, results)
    save_as_pickle(save_path_intervention, resutls_intervention)
    # lang_ratios
    save_path_lang_ratios = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qa/{model_type}/lang_ratio/lang_ratios_add_subtracted_vectors_to_last_two_layers_only.pkl'
    save_as_pickle(save_path_lang_ratios, lang_ratios_final)
    
    # print results.
    print(f'{model_type}\n{resutls_intervention}')
    
    # release memory.
    del model
    torch.cuda.empty_cache()


""" for output """
print(f'q_num: {qa_num}')
print('===============================================================================')
print(f'intervened_layers: None')
print(resutls_intervention)