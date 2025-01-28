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
activation_types = ["abs", "product"]
norm_type = "no"
intervention_num = 15000

for L2, model_name in model_names.items():
    """ get curpus and models """

    """ tatoeba translation corpus """
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    num_sentences = 2000
    dataset = dataset.select(range(num_sentences))
    tatoeba_data = []
    for item in dataset:
        # check if there are empty sentences.
        if item['translation'][L1] != '' and item['translation'][L2] != '':
            tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
    # tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]
    tatoeba_data_len = len(tatoeba_data)
    """
    baseとして、対訳関係のない1文ずつのペアを作成
    (L1(en)はhttps://huggingface.co/datasets/agentlans/high-quality-english-sentences,
    L2はtatoebaの該当データを使用)
    """
    random_data = []
    # L1(en)
    en_base_ds = load_dataset("agentlans/high-quality-english-sentences")
    random_data_en = en_base_ds["train"][:num_sentences]
    en_base_ds_idx = 0
    for item in dataset:
        random_data.append((random_data_en["text"][en_base_ds_idx], item["translation"][L2]))
        en_base_ds_idx += 1

    """ model and device configs """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = model_names[L2]
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    """
    get act_pattenrs as cos_sim (with no intervention).
    """
    # get activation list
    act_patterns = get_act_patterns(model, tokenizer, device, tatoeba_data)
    act_patterns_baseline = get_act_patterns(model, tokenizer, device, random_data)
    # plot activation patterns.
    activation_patterns_lineplot(act_patterns, act_patterns_baseline, L2, activation_type, "no")

    for activation_type in activation_types: # abs, product両方のAP上位を試す.
        """
        get act_patterns as cos_sim (with high AP neurons intervention).
        """
        # unfreeze AP_list.
        pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/sorted_neurons_{L2}.pkl"
        sorted_neurons_AP = unfreeze_pickle(pkl_file_path)

        """ どのくらい介入するか(intervention_num) """
        sorted_neurons_AP = sorted_neurons_AP[:intervention_num]
        # baseline
        sorted_neurons_AP_baseline = random.sample(sorted_neurons_AP[intervention_num+1:], len(sorted_neurons_AP[intervention_num+1:]))
        sorted_neurons_AP_baseline = sorted_neurons_AP_baseline[:intervention_num]

        """ deactivate high AP neurons. """
        # get activation list
        act_patterns = get_act_patterns_with_edit_activation(model, tokenizer, device, sorted_neurons_AP, tatoeba_data)
        act_patterns_baseline = get_act_patterns_with_edit_activation(model, tokenizer, device, sorted_neurons_AP, random_data)
        # plot activation patterns.
        activation_patterns_lineplot(act_patterns, act_patterns_baseline, L2, activation_type, "yes")

        """ deactivate baseline neurons. """
        # get activation list
        act_patterns = get_act_patterns_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, tatoeba_data)
        act_patterns_baseline = get_act_patterns_with_edit_activation(model, tokenizer, device, sorted_neurons_AP_baseline, random_data)
        # plot activation patterns.
        activation_patterns_lineplot(act_patterns, act_patterns_baseline, L2, activation_type, "baseline")

        print(f"intervention_num: {intervention_num} <- completed.")

    # delete model and some cache
    del model
    torch.cuda.empty_cache()