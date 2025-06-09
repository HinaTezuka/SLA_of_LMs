import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons")
import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from funcs import (
    get_act_patterns,
    get_act_patterns_inputs_to_down_proj,
    get_act_patterns_with_edit_activation,
    activation_patterns_lineplot,
    save_as_pickle,
    unfreeze_pickle,
)

L1 = "en" # fix L1 to English.
""" model configs """
model_names = ['CohereForAI/aya-expanse-8b', 'meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'microsoft/phi-4', 'Qwen/Qwen3-8B']
device = "cuda" if torch.cuda.is_available() else "cpu"
n_list = [100, 1000, 3000, 5000] # patterns of intervention_num
score_types = ['cos_sim', 'L2_dis']
langs = ['ja', 'nl', 'ko', 'it', 'fr', 'ru', 'vi']
is_reverses = [False, True]

for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya' if 'aya' in model_name else 'phi4'
    if model_type == 'phi4':
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    """ parallel data """
    for L2 in langs:
        """ tatoeba translation corpus """
        dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
        # # select first 2000 sentences.
        total_sentence_num = 2000 if L2 == "ko" else 5000
        num_sentences = 1000
        dataset = dataset.select(range(total_sentence_num))

        # same semantics sentence pairs: test split.
        tatoeba_data = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_multi_test.pkl")

        """ non-translation pair for baseline. """
        random_data = []
        if L2 == "ko": # koreanはデータ数が足りない
            dataset2 = load_dataset("tatoeba", lang1=L1, lang2="ja", split="train").select(range(5000))
        for sentence_idx, item in enumerate(dataset):
            if sentence_idx == num_sentences: break
            if L2 == "ko" and dataset2['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
                random_data.append((dataset2["translation"][num_sentences+sentence_idx][L1], item["translation"][L2])) 
            elif L2 != "ko" and dataset['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
                random_data.append((dataset["translation"][num_sentences+sentence_idx][L1], item["translation"][L2]))

        for is_reverse in is_reverses:
            for score_type in score_types:
                if score_type == "cos_sim":
                    act_patterns = get_act_patterns(model, model_type, tokenizer, device, tatoeba_data)
                    act_patterns_baseline = get_act_patterns(model, model_type, tokenizer, device, random_data)
                    activation_patterns_lineplot(act_patterns, act_patterns_baseline, L2, None, model_type, score_type, "normal")
                for intervention_num in n_list:
                    """
                    get act_patterns as cos_sim (with high AP neurons intervention).
                    """
                    if is_reverse:
                        # prepare type-2 Transfer Neurons.
                        save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
                        sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
                        if model_type == 'phi4':
                            sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(30, 40)]]
                        else:
                            sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
                    else:
                        # type-1 neurons.
                        save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
                        sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)

                    """ どのくらい介入するか(intervention_num) """
                    sorted_neurons_AP = sorted_neurons[:intervention_num]
                    # baseline
                    random.seed(42)
                    sorted_neurons_AP_baseline = random.sample(sorted_neurons[intervention_num:], intervention_num)

                    """ deactivate high AP neurons. """
                    # get activation list
                    act_patterns_parallel = get_act_patterns_with_edit_activation(model, tokenizer, device, sorted_neurons_AP, tatoeba_data, model_type)
                    act_patterns_random = get_act_patterns_with_edit_activation(model, tokenizer, device, sorted_neurons_AP, random_data, model_type)
                    # plot activation patterns.
                    activation_patterns_lineplot(act_patterns_parallel, act_patterns_random, L2, intervention_num, model_type, score_type, "yes", is_reverse)

                    """ deactivate baseline neurons. """
                    # get activation list
                    act_patterns_parallel = get_act_patterns_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, tatoeba_data, model_type)
                    act_patterns_random = get_act_patterns_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, random_data, model_type)
                    # plot activation patterns.
                    activation_patterns_lineplot(act_patterns_parallel, act_patterns_random, L2, intervention_num, model_type, score_type, "baseline", is_reverse)
    
    # clean cache.
    del model
    torch.cuda.empty_cache()