import sys
import dill as pickle
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from cld3_funcs import (
    project_hidden_emb_to_vocab,
    get_hidden_states_with_edit_activation,
    layerwise_lang_stats,
    layerwise_lang_distribution,
    plot_lang_distribution,
    print_tokens,
    unfreeze_pickle,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", "CohereForAI/aya-expanse-8b"]
layer_num = 32
score_types = ["cos_sim", "L2_dis"]
norm_type = "no"
langs = ["ja", "nl", "ko", "it"]

""" prepare mkqa """
qa = load_dataset('apple/mkqa')['train']
qa = qa.shuffle(seed=42)
score_type = 'cos_sim'
intervention_num = 1000
qa_num = 100
is_reverses = ["normal", False, True]

for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    for is_reverse in is_reverses:
        for score_type in score_types:
            for L2 in langs:
                ratio_en = np.zeros((layer_num, qa_num)) # en token_numの数を保存しておくarray(後で平均を計算する用)
                ratio_L2 = np.zeros((layer_num, qa_num)) # L2 token_numの数を保存しておくarray(後で平均を計算する用)
                ratio_en_baseline = np.zeros((layer_num, qa_num))
                ratio_L2_baseline = np.zeros((layer_num, qa_num))

                for i in range(len(qa['queries'])):
                    if i == qa_num: break

                    """ prepare inputs. """
                    q = qa['queries'][i][L2] # question
                    # make prompt.
                    if L2 == 'ja': prompt = f'{q}? 答え: '
                    elif L2 == 'nl': prompt = f'{q}? Antwoord: '
                    elif L2 == 'ko': prompt = f'{q}? 답변: '
                    elif L2 == 'it': prompt = f'{q}? Risposta: '
                    elif L2  == 'en': prompt = f'{q}? Answer: '
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)

                    if is_reverse == "normal":
                        """ get hidden states of all layers (without intervention). """
                        last_token_index = inputs["input_ids"].shape[1] - 1 # index corrensponding to the last token of the inputs.
                        # run inference
                        with torch.no_grad():
                            output = model(**inputs, output_hidden_states=True)
                        # ht
                        all_hidden_states = output.hidden_states[1:] # exclude emb_layer.
                    else: # get hs with intervention.
                        if not is_reverse:
                            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
                            neurons = unfreeze_pickle(path)
                            neurons = [neuron for neuron in neurons if neuron[0] in [ _ for _ in range(20)]]
                        else:
                            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/aya/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
                            neurons = unfreeze_pickle(path)
                            neurons = [neuron for neuron in neurons if neuron[0] in [ _ for _ in range(20, 32)]]
                        # intervention neurons
                        neurons = neurons[:intervention_num]
                        # baseline neurons
                        random.seed(42)
                        neurons_baseline = random.sample(neurons[intervention_num+1:], intervention_num)

                        all_hidden_states = get_hidden_states_with_edit_activation(model, inputs, neurons)
                        all_hidden_states_baseline = get_hidden_states_with_edit_activation(model, inputs, neurons_baseline)

                    """ get topk tokens per each hidden state. """
                    top_k = 100 # token nums to decode.
                    if score_type == "cos_sim" and is_reverse == "normal":
                        # get hidden state of the layer(last token only).
                        tokens_dict = project_hidden_emb_to_vocab(model, tokenizer, all_hidden_states, last_token_index, top_k=top_k)

                    if is_reverse != "normal":
                        tokens_dict_intervention = project_hidden_emb_to_vocab(model, tokenizer, all_hidden_states, last_token_index, top_k=top_k)
                        tokens_dict_baseline = project_hidden_emb_to_vocab(model, tokenizer, all_hidden_states_baseline, last_token_index, top_k=top_k)

                    # non-intervention.
                    if score_type == "cos_sim" and is_reverse == "normal":
                        # normal
                        lang_stats_normal = layerwise_lang_stats(tokens_dict, L2)
                        for layer_i in range(layer_num):
                            ratio_en[layer_i, i] = lang_stats_normal[layer_i]['en'] # i: question_idx.
                            ratio_L2[layer_i, i] = lang_stats_normal[layer_i][L2] 

                    if is_reverse != "normal":
                        # high APs intervention
                        lang_stats_intervention = layerwise_lang_stats(tokens_dict_intervention, L2)
                        for layer_i in range(layer_num):
                            ratio_en[layer_i, i] = lang_stats_intervention[layer_i]['en']
                            ratio_L2[layer_i, i] = lang_stats_intervention[layer_i][L2]

                        # baseline intervention
                        lang_stats_baseline = layerwise_lang_stats(tokens_dict_baseline, L2)
                        for layer_i in range(layer_num):
                            ratio_en_baseline[layer_i, i] = lang_stats_baseline[layer_i]['en']
                            ratio_L2_baseline[layer_i, i] = lang_stats_baseline[layer_i][L2]

                lang_stats_normal = defaultdict(lambda: defaultdict(int))
                lang_stats_intervention = defaultdict(lambda: defaultdict(int))
                lang_stats_baseline = defaultdict(lambda: defaultdict(int))
                for layer_i in range(layer_num):
                    if score_type == "cos_sim" and is_reverse == "normal":
                        lang_stats_normal[layer_i]['en'] = int(np.mean(ratio_en[layer_i, :]))
                        lang_stats_normal[layer_i][L2] = int(np.mean(ratio_en[layer_i, :]))
                        lang_stats_normal[layer_i]['total_count'] = lang_stats_normal[layer_i]['en'] + lang_stats_normal[layer_i][L2]
                    else: # intervention.
                        lang_stats_intervention[layer_i]['en'] = int(np.mean(ratio_en[layer_i, :]))
                        lang_stats_intervention[layer_i][L2] = int(np.mean(ratio_en[layer_i, :]))
                        lang_stats_intervention[layer_i]['total_count'] = lang_stats_intervention[layer_i]['en'] + lang_stats_intervention[layer_i][L2]
                        lang_stats_baseline[layer_i]['en'] = int(np.mean(ratio_en_baseline[layer_i, :]))
                        lang_stats_baseline[layer_i][L2] = int(np.mean(ratio_en_baseline[layer_i, :]))
                        lang_stats_baseline[layer_i]['total_count'] = lang_stats_baseline[layer_i]['en'] + lang_stats_baseline[layer_i][L2]


                """ visualization (intervention). """
                if score_type == "cos_sim" and is_reverse == "normal":
                    lang_distribution_normal = layerwise_lang_distribution(lang_stats_normal, L2)
                    plot_lang_distribution(lang_distribution_normal, "normal", model_type, intervention_num, L2)
                if is_reverse != "normal":
                    # intervention
                    lang_distribution_intervention = layerwise_lang_distribution(lang_stats_intervention, L2)
                    plot_lang_distribution(lang_distribution_intervention, score_type, model_type, intervention_num, L2, is_reverse)
                    # intervention baseline
                    lang_distribution_baseline = layerwise_lang_distribution(lang_stats_baseline, L2)
                    plot_lang_distribution(lang_distribution_baseline, score_type, model_type, intervention_num, L2, is_reverse)
                
    # delete caches
    del model
    torch.cuda.empty_cache()
    