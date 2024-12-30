"""
calc acc of blimp_task with neuron intervention.
using: act_sum_shared.
発火値の合計から上位nコをdeactivateした上で BLiMP の精度を再測定.
"""
import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons")
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention")
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/hidden_state_sim")
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/hidden_state_sim/AUC")
import dill as pickle
from collections import defaultdict

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from intervention_funcs import (
    save_as_pickle,
    unfreeze_pickle,
    delete_specified_keys_from_act_sum_dict,
    display_activation_values,
    eval_JBLiMP_with_edit_activation,
    get_complement,
    has_overlap,
    delete_overlaps,
)

""" parameters """
L2 = "ja"
active_THRESHOLD = 0.01
activation_type = "abs"
norm_type = "no"

""" act_sum_base_dict(non translation pair): <= 非対訳ペアに発火しているそれぞれのtypeのneurons """
pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/non_same_semantics/act_sum/{active_THRESHOLD}_th/en_{L2}.pkl"
act_sum_dict = unfreeze_pickle(pkl_file_path)
# それぞれのneuronsの発火値の合計（dict)を取得
act_sum_shared = act_sum_dict["shared"]
act_sum_L1_or_L2 = act_sum_dict["L1_or_L2"]
act_sum_L1_specific = act_sum_dict["L1_specific"]
act_sum_L2_specific = act_sum_dict["L2_specific"]

""" shared_neuronsのうち、AP上位nコ """
pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/sorted_neurons_{L2}.pkl"
sorted_neurons_AP = unfreeze_pickle(pkl_file_path)

""" shared neurons(非対訳ペア) """
non_translation_shared = []
for layer_idx, neurons in act_sum_shared.items():
    for neuron_idx in neurons.keys():
        non_translation_shared.append((layer_idx, neuron_idx))
non_translation_shared = sorted(non_translation_shared, key=lambda x: act_sum_shared[x[0]][x[1]], reverse=True)

""" L1 or L2に発火したニューロン(非対訳ペア) """
layer_neuron_list_L1_or_L2 = []
for layer_idx, neurons in act_sum_L1_or_L2.items():
    for neuron_idx in neurons.keys():
        layer_neuron_list_L1_or_L2.append((layer_idx, neuron_idx))
layer_neuron_list_L1_or_L2 = sorted(layer_neuron_list_L1_or_L2, key=lambda x: act_sum_L1_or_L2[x[0]][x[1]], reverse=True)

""" L1のみに発火しているニューロン(非対訳ペア) """
layer_neuron_list_L1_specific = []
for layer_idx, neurons in act_sum_L1_specific.items():
    for neuron_idx in neurons.keys():
        layer_neuron_list_L1_specific.append((layer_idx, neuron_idx))
layer_neuron_list_L1_specific = sorted(layer_neuron_list_L1_specific, key=lambda x: act_sum_L1_specific[x[0]][x[1]], reverse=True)

""" どのくらい介入するか(n) """
intervention_num = 10000
sorted_neurons_AP = sorted_neurons_AP[:intervention_num]
non_translation_shared = non_translation_shared[:intervention_num]
layer_neuron_list_L1_or_L2 = layer_neuron_list_L1_or_L2[:intervention_num]
layer_neuron_list_L1_specific = layer_neuron_list_L1_specific[:intervention_num]

if __name__ == "__main__":
    """ neuron intervention (発火値の改竄実験)"""

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

    model_name = model_names[L2]
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # CSV保存用dir
    dir_path = f"n_{intervention_num}"

    """ same semantics shared neurons """
    result_AP = eval_JBLiMP_with_edit_activation(model, model_name, tokenizer, sorted_neurons_AP)
    print(f"result_AP(topN_on_AP): {result_AP}")
    df_main = pd.DataFrame(result_AP)
    # calc overall
    overall_accuracy_main = df_main.groupby('Model')['Accuracy'].mean().reset_index()
    overall_accuracy_main.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    # save as csv
    df_main.to_csv(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/csv/llama3/AUC/jblimp/AP/{dir_path}/en_{L2}.csv", index=False)

    """ non same semantics shared neurons """
    result_shared_non_translation = eval_JBLiMP_with_edit_activation(model, model_name, tokenizer, non_translation_shared)
    print(f"result_shared_non_translation: {result_shared_non_translation}")
    df_shared_non_translation = pd.DataFrame(result_shared_non_translation)
    overall_accuracy_shared_non_translation = df_shared_non_translation.groupby('Model')['Accuracy'].mean().reset_index()
    overall_accuracy_shared_non_translation.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    df_shared_non_translation.to_csv(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/csv/llama3/AUC/jblimp/non_translation/{dir_path}/en_{L2}.csv", index=False)

    """ L1 or L2 """
    result_comp_L1_or_L2 = eval_JBLiMP_with_edit_activation(model, model_name, tokenizer, layer_neuron_list_L1_or_L2)
    print(f"result_comp_L1_or_L2: {result_comp_L1_or_L2}")
    df_comp_L1_or_L2 = pd.DataFrame(result_comp_L1_or_L2)
    overall_accuracy_comp_L1_or_L2 = df_comp_L1_or_L2.groupby('Model')['Accuracy'].mean().reset_index()
    overall_accuracy_comp_L1_or_L2.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    df_comp_L1_or_L2.to_csv(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/csv/llama3/AUC/jblimp/L1_or_L2/{dir_path}/en_{L2}.csv", index=False)

    """ L1 Specific """
    result_comp_L1_specific = eval_JBLiMP_with_edit_activation(model, model_name, tokenizer, layer_neuron_list_L1_specific)
    print(f"result_comp_L1_specific: {result_comp_L1_specific}")
    df_comp_L1_specific = pd.DataFrame(result_comp_L1_specific)
    overall_accuracy_comp_L1_specific = df_comp_L1_specific.groupby('Model')['Accuracy'].mean().reset_index()
    overall_accuracy_comp_L1_specific.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    df_comp_L1_specific.to_csv(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/csv/llama3/AUC/jblimp/L1_specific/{dir_path}/en_{L2}.csv", index=False)

    """ print OVERALLs and Meta Info"""
    print("============================ OVER ALL ============================")
    print(f"AP: {overall_accuracy_main}")
    print(f"non_translation_shared: {overall_accuracy_shared_non_translation}")
    print(f"L1_or_L2: {overall_accuracy_comp_L1_or_L2}")
    print(f"L1_specific: {overall_accuracy_comp_L1_specific}")

    print("============================ META INFO ============================")
    print(f"L2: {L2}")
    print(f"intervention num: {intervention_num}")
    print(f"intervention_num percentage: {float(intervention_num / 14336 * 32)*100} %.")
    print("completed. saved to csv.")
