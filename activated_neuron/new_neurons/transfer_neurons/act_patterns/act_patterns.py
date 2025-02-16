import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/activation_patterns")
import random
import dill as pickle

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from funcs import (
    get_act_patterns,
    get_act_patterns_with_edit_activation,
    activation_patterns_lineplot,
    save_as_pickle,
    unfreeze_pickle,
)

L1 = "en"
""" model configs """
# LLaMA-3
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B"
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}
# params
norm_type = "no"
intervention_num = 1000

for L2, model_name in model_names.items():
    """ get curpus and models """

    """ tatoeba translation corpus """
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # select first 2000 sentences
    total_sentence_num = 2000 if L2 == "ko" else 5000
    num_sentences = 2000
    dataset = dataset.select(range(total_sentence_num))
    tatoeba_data = []
    for sentence_idx, item in enumerate(dataset):
        if sentence_idx == num_sentences: break
        # check if there are empty sentences.
        if item['translation'][L1] != '' and item['translation'][L2] != '':
            tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
    tatoeba_data_len = len(tatoeba_data)

    """
    baseとして、対訳関係のない1文ずつのペアを作成
    """
    random_data = []
    if L2 == "ko": # koreanはデータ数が足りない
        dataset2 = load_dataset("tatoeba", lang1=L1, lang2="ja", split="train").select(range(5000))
    for sentence_idx, item in enumerate(dataset):
        if sentence_idx == num_sentences: break
        if L2 == "ko" and dataset2['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
            random_data.append((dataset2["translation"][num_sentences+sentence_idx][L1], item["translation"][L2])) 
        elif L2 != "ko" and dataset['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
            random_data.append((dataset["translation"][num_sentences+sentence_idx][L1], item["translation"][L2]))

    """ model and device configs """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    """
    get act_patterns as cos_sim (with high AP neurons intervention).
    """
    # unfreeze AP_list.
    save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/ap_lang_specific/sorted_neurons_{L2}.pkl"
    sorted_neurons_AP = unfreeze_pickle(save_path_sorted_neurons)

    """ どのくらい介入するか(intervention_num) """
    sorted_neurons_AP = sorted_neurons_AP[:intervention_num] + sorted_neurons_AP[-intervention_num:]
    # baseline
    # sorted_neurons_AP_baseline = random.sample(sorted_neurons_AP[intervention_num+1:], len(sorted_neurons_AP[intervention_num+1:]))
    # sorted_neurons_AP_baseline = sorted_neurons_AP_baseline[:intervention_num]

    """ deactivate high AP neurons. """
    # get activation list
    act_patterns = get_act_patterns_with_edit_activation(model, tokenizer, device, sorted_neurons_AP, tatoeba_data)
    act_patterns_baseline = get_act_patterns_with_edit_activation(model, tokenizer, device, sorted_neurons_AP, random_data)
    # plot activation patterns.
    activation_patterns_lineplot(act_patterns, act_patterns_baseline, L2, intervention_num, "yes")

    """ deactivate baseline neurons. """
    # # get activation list
    # act_patterns = get_act_patterns_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, tatoeba_data)
    # act_patterns_baseline = get_act_patterns_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, random_data)
    # # plot activation patterns.
    # activation_patterns_lineplot(act_patterns, act_patterns_baseline, L2, intervention_num, "baseline")

    print(f"intervention_num: {intervention_num} <- completed.")

    # delete model and some cache
    del model
    torch.cuda.empty_cache()