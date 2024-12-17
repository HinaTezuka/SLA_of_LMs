import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")
sys.path.append("/home/s2410121/proj_LA/activated_neuron/neuron_intervention")
import dill as pickle
from collections import defaultdict

import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from neuron_intervention_funcs import (
    eval_BLiMP_with_edit_activation,
    get_complement,
    has_overlap,
    delete_overlaps,
)

L2 = "ja"
""" load pkl_file(act_sum_dict) """
pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/pickles/act_sum/tatoeba_0_th/act_sum_dict/act_sum_dict_en_{L2}_tatoeba_0_th.pkl"
with open(pkl_file_path, "rb") as f:
    act_sum_dict = pickle.load(f)
print("unfolded pickle: act_sum_dict")

# それぞれのneuronsの発火値の合計（dict)を取得
act_sum_shared = act_sum_dict["shared"] # 非対訳ペアに発火しているshared neuronsも含む。
act_sum_L1_or_L2 = act_sum_dict["L1_or_L2"]
act_sum_L1_specific = act_sum_dict["L1_specific"]
act_sum_L2_specific = act_sum_dict["L2_specific"]

""" load pkl_file(act_freq_base_dict): 非対訳ペアに発火している shared neruons <- 対訳ペアのみに対して発火しているshared neuronsをとるため """
pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/pickles/base/act_freq/tatoeba_0_th/act_freq_dict_en_{L2}_tatoeba_0_th.pkl"
with open(pkl_file_path, "rb") as f:
    act_freq_base_dict = pickle.load(f)
print("unfolded pickle: act_freq_base_dict")

# shared neuronsのうち、非対訳ペアに対して発火したneuronsを取得 <- dict: {layer_idx: neuron_idx}
freq_base_shared = act_freq_base_dict["shared_neurons"]

"""
shared neurons全体の集合から、非対訳ペアに対してよく発火しているshared neuronsを除去
（対訳ペア:同じ意味表現に発火しているshared neuronsのみを抽出）
"""
# 非対訳ペア2000組のうち、(THRESHOLD)割以上発火しているshared neuronsを取得
# shared_neurons_non_translations = []  # list of tuples: [(layer_idx, neuron_idx)]
THRESHOLD = 0
# THRESHOLD_corpus = 2000 * THRESHOLD  #

# # 非対訳ペアに該当するニューロンを収集
# for layer_idx, neurons in freq_base_shared.items():
#     for neuron_idx, act_freqency in neurons.items():
#         if act_freqency > THRESHOLD_corpus:
#             shared_neurons_non_translations.append((layer_idx, neuron_idx))

# # 削除対象のニューロンを記録
# keys_to_remove = []
# for layer_idx, neurons in act_sum_shared.items():
#     for neuron_idx in neurons.keys():  # 辞書に直接ループ
#         if (layer_idx, neuron_idx) in shared_neurons_non_translations:
#             keys_to_remove.append((layer_idx, neuron_idx))

# # 一括削除処理
# for layer_idx, neuron_idx in keys_to_remove:
#     del act_sum_shared[layer_idx][neuron_idx]
#     # サブ辞書が空の場合、親キーを削除
#     if not act_sum_shared[layer_idx]:
#         del act_sum_shared[layer_idx]

""" shared_neurons_ONLYをpickle fileとして保存(初回のみ) """
# pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/pickles/shared_same_semantics/shared_ONLY_dict_en_{L2}_tatoeba_0_th.pkl"
# # directoryを作成（存在しない場合のみ)
# os.makedirs(os.path.dirname(pkl_file_path), exist_ok=True)
# with open(pkl_file_path, "wb") as f:
#     pickle.dump(act_sum_shared, f)
# print("pickle file saved.")

""" shared_ONLY_dictをロード（同じ意味表現にのみ発火しているshared neurons） """
pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/pickles/shared_same_semantics/shared_ONLY_dict_en_{L2}_tatoeba_0_th.pkl"
with open(pkl_file_path, "rb") as f:
    act_sum_shared = pickle.load(f)
print("unfolded pickle: act_same_semantics")

count_shared_ONLY = 0
for layer_idx in act_sum_shared.keys():
    count_shared_ONLY += len(act_sum_shared[layer_idx])

"""
list[(layer_idx, neuron_idx), ...] <= 介入実験用
listはact_sumを軸に降順にソート
同じ意味表現（対訳ペア）のみにたいして発火しているshared neurons
"""
shared_same_semantics = [] # shared neurons [(layer_idx, neuron_idx), ...]
for layer_idx, neurons in act_sum_shared.items():
    for neuron_idx in neurons.keys():
        shared_same_semantics.append((layer_idx, neuron_idx))
shared_same_semantics = sorted(shared_same_semantics, key=lambda x: act_sum_shared[x[0]][x[1]], reverse=True)

"""
非対訳ペアにたいして発火しているshared neurons
"""
non_translation_shared = []
for layer_idx, neurons in freq_base_shared.items():
    for neuron_idx in neurons.keys():
        non_translation_shared.append((layer_idx, neuron_idx))
non_translation_shared = sorted(non_translation_shared, key=lambda x: freq_base_shared[x[0]][x[1]], reverse=True)

""" 作成したlayer_neuron_listの補集合を作成(発火している/していない関係なく) """
# 全層数と各層のニューロン数
num_layers = 32
num_neurons_per_layer = 14336
all_layers = range(num_layers)
all_neurons = range(num_neurons_per_layer)
# 補集合の生成
complement_list = get_complement(all_layers, all_neurons, shared_same_semantics)

""" (activate)したニューロンの中から、layer_neuron_listの補集合を作成: <- つまり、L1 or L2に発火したニューロン """
layer_neuron_list_L1_or_L2 = []
for layer_idx, neurons in act_sum_L1_or_L2.items():
    for neuron_idx in neurons.keys():
        layer_neuron_list_L1_or_L2.append((layer_idx, neuron_idx))
# shared_same_semanticsとかぶっている要素は削除
layer_neuron_list_L1_or_L2 = delete_overlaps(layer_neuron_list_L1_or_L2, shared_same_semantics)
# sort
layer_neuron_list_L1_or_L2 = sorted(layer_neuron_list_L1_or_L2, key=lambda x: act_sum_L1_or_L2[x[0]][x[1]], reverse=True)

""" L1のみに発火しているニューロンの中から、layer_neuron_listを作成 """
layer_neuron_list_L1_specific = []
for layer_idx, neurons in act_sum_L1_specific.items():
    for neuron_idx in neurons.keys():
        layer_neuron_list_L1_specific.append((layer_idx, neuron_idx))
# shared_same_semanticsとかぶっている要素は削除
layer_neuron_list_L1_specific = delete_overlaps(layer_neuron_list_L1_specific, shared_same_semantics)
# sort
layer_neuron_list_L1_specific = sorted(layer_neuron_list_L1_specific, key=lambda x: act_sum_L1_specific[x[0]][x[1]], reverse=True)


""" どのくらい介入するか(n) """
intervention_num = count_shared_ONLY
# intervention_num = 20000
shared_same_semantics = shared_same_semantics[:intervention_num]
non_translation_shared = non_translation_shared[:intervention_num]
complement_list = complement_list[:intervention_num]
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
    dir_path = "all"
    # dir_path = f"n_{intervention_num}"

    """ same semantics shared neurons """
    # result_main = eval_BLiMP_with_edit_activation(model, model_name, tokenizer, shared_same_semantics)
    # print(f"result_main: {result_main}")
    # df_main = pd.DataFrame(result_main)
    # # calc overall
    # overall_accuracy_main = df_main.groupby('Model')['Accuracy'].mean().reset_index()
    # overall_accuracy_main.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    # # save as csv
    # df_main.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/shared/{dir_path}/llama3_en_{L2}_shared_ONLY.csv", index=False)

    """ non same semantics shared neurons """
    # result_shared_non_translation = eval_BLiMP_with_edit_activation(model, model_name, tokenizer, non_translation_shared)
    # print(f"result_shared_non_translation: {result_shared_non_translation}")
    # df_shared_non_translation = pd.DataFrame(result_shared_non_translation)
    # overall_accuracy_shared_non_translation = df_shared_non_translation.groupby('Model')['Accuracy'].mean().reset_index()
    # overall_accuracy_shared_non_translation.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    # df_shared_non_translation.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/shared/{dir_path}/llama3_en_{L2}_shared_non_translation.csv", index=False)

    """ normal COMP """
    # result_comp = eval_BLiMP_with_edit_activation(model, model_name, tokenizer, complement_list)
    # print(f"result_comp: {result_comp}")
    # df_comp = pd.DataFrame(result_comp)
    # overall_accuracy_comp = df_comp.groupby('Model')['Accuracy'].mean().reset_index()
    # overall_accuracy_comp.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    # df_comp.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/normal_COMP/{dir_path}/llama3_en_{L2}_COMP.csv", index=False)

    """ L1 or L2 """
    result_comp_L1_or_L2 = eval_BLiMP_with_edit_activation(model, model_name, tokenizer, layer_neuron_list_L1_or_L2)
    print(f"result_comp_L1_or_L2: {result_comp_L1_or_L2}")
    df_comp_L1_or_L2 = pd.DataFrame(result_comp_L1_or_L2)
    overall_accuracy_comp_L1_or_L2 = df_comp_L1_or_L2.groupby('Model')['Accuracy'].mean().reset_index()
    overall_accuracy_comp_L1_or_L2.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    df_comp_L1_or_L2.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/L1_or_L2/{dir_path}/llama3_en_{L2}_L1_or_L2.csv", index=False)

    """ L1 Specific """
    result_comp_L1_specific = eval_BLiMP_with_edit_activation(model, model_name, tokenizer, layer_neuron_list_L1_specific)
    print(f"result_comp_L1_specific: {result_comp_L1_specific}")
    df_comp_L1_specific = pd.DataFrame(result_comp_L1_specific)
    overall_accuracy_comp_L1_specific = df_comp_L1_specific.groupby('Model')['Accuracy'].mean().reset_index()
    overall_accuracy_comp_L1_specific.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    df_comp_L1_specific.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/L1_specific/{dir_path}/llama3_en_{L2}_L1_specific.csv", index=False)

    """ print OVERALL and Meta Info"""
    # print(f"overall_accuracy_main: {overall_accuracy_main}")
    # print(f"overall_accuracy_shared_non_translation: {overall_accuracy_shared_non_translation}")
    # print(f"overall_accuracy_comp: {overall_accuracy_comp}")
    print(f"overall_accuracy_comp_L1_or_L2: {overall_accuracy_comp_L1_or_L2}")
    print(f"overall_accuracy_comp_L1_specific: {overall_accuracy_comp_L1_specific}")

    print("============================ META INFO ============================")
    print(f"L2: {L2}")
    print(f"intervention num: {intervention_num}")
    print(f"intervention_num percentage: {float(intervention_num / count_shared_ONLY)} %.")
    print(f"THRESHOLD: {THRESHOLD}")
    print(f"count_shared_ONLY(same semantics): {count_shared_ONLY}")
    print("completed. saved to csv.")
