import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")
sys.path.append("/home/s2410121/proj_LA/activated_neuron/neuron_intervention")
import dill as pickle

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import transformers

from baukit import Trace, TraceDict
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from neuron_intervention_funcs import evaluate_sentence_pair_with_edit_activation, get_complement, has_overlap

""" load pkl_file(act_sum_dict) """
pkl_file_path = "/home/s2410121/proj_LA/activated_neuron/pickles/act_sum/tatoeba_0_th/act_sum_dict/act_sum_dict_en_ja_tatoeba_0_th.pkl"
with open(pkl_file_path, "rb") as f:
    act_sum_dict = pickle.load(f)
print("unfolded pickle: act_sum_dict")

# それぞれのneuronsの発火値の合計（dict)を取得
act_sum_shared = act_sum_dict["shared"]
act_sum_L1_or_L2 = act_sum_dict["L1_or_L2"]
act_sum_L1_specific = act_sum_dict["L1_specific"]
act_sum_L2_specific = act_sum_dict["L2_specific"]

""" load pkl_file(act_freq_base_dict) <- 対訳ペアのみに対して発火しているshared neuronsをとるため """
pkl_file_path = "/home/s2410121/proj_LA/activated_neuron/pickles/base/act_freq/tatoeba_0_th/act_freq_dict_en_ja_tatoeba_0_th.pkl"
with open(pkl_file_path, "rb") as f:
    act_freq_base_dict = pickle.load(f)
print("unfolded pickle: act_freq_base_dict")

# shared neuronsのうち、非対訳ペアに対して発火したneuronsを取得 <- dict: {layer_idx: neuron_idx}
freq_base_shared = act_freq_base_dict["shared_neurons"]

c_before = 0
for layer_idx in act_sum_shared.keys():
    c_before += len(act_sum_shared[layer_idx])
print(c_before)

"""
shared neurons全体の集合から、非対訳ペアに対してよく発火しているshared neuronsを除去
（対訳ペア:同じ意味表現に発火しているshared neuronsのみを抽出）
"""
# 非対訳ペア2000組のうち、(THRESHOLD)割以上発火しているshared neuronsを取得
shared_neurons_non_translations = []  # list of tuples: [(layer_idx, neuron_idx)]
THRESHOLD = 0
THRESHOLD_corpus = 2000 * THRESHOLD  # 1割以上

# 非対訳ペアに該当するニューロンを収集
for layer_idx, neurons in freq_base_shared.items():
    for neuron_idx, act_freqency in neurons.items():
        if act_freqency > THRESHOLD_corpus:
            shared_neurons_non_translations.append((layer_idx, neuron_idx))

# 削除対象のニューロンを記録
keys_to_remove = []
for layer_idx, neurons in act_sum_shared.items():
    for neuron_idx in neurons.keys():  # 辞書に直接ループ
        if (layer_idx, neuron_idx) in shared_neurons_non_translations:
            keys_to_remove.append((layer_idx, neuron_idx))

# 一括削除処理
for layer_idx, neuron_idx in keys_to_remove:
    del act_sum_shared[layer_idx][neuron_idx]
    # サブ辞書が空の場合、親キーを削除
    if not act_sum_shared[layer_idx]:
        del act_sum_shared[layer_idx]

c_after = 0
for layer_idx in act_sum_shared.keys():
    c_after += len(act_sum_shared[layer_idx])
# print(c_after)
# sys.exit()
"""
list [(layer_idx, neuron_idx, sum), (layer_idx, neuron_idx, sum_of_act_values), ...] <= ちゃんと取れているか確認用
listはact_sumを軸に降順にソート
"""
# ちゃんと降順にとれている確認用に、(layer_idx, neuron_idx, sum_of_act_values)を表示
# tuple_list = []
# for layer_idx, neurons in act_sum_shared.items():
#     for neuron_idx, act_sum in neurons.items():
#         tuple_list.append((layer_idx, neuron_idx, act_sum))
# tuple_list = sorted(tuple_list, key=lambda x: x[2], reverse=True)

"""
list[(layer_idx, neuron_idx), ...] <= 介入実験用
listはact_sumを軸に降順にソート
"""
layer_neuron_list = [] # shared neurons [(layer_idx, neuron_idx), ...]
for layer_idx, neurons in act_sum_shared.items():
    for neuron_idx in neurons.keys():
        layer_neuron_list.append((layer_idx, neuron_idx))
layer_neuron_list = sorted(layer_neuron_list, key=lambda x: act_sum_shared[x[0]][x[1]], reverse=True)
# print(layer_neuron_list[:10])

""" 作成したlayer_neuron_listの補集合を作成(発火している/していない関係なく) """
# 全層数と各層のニューロン数
num_layers = 32
num_neurons_per_layer = 14336
all_layers = range(num_layers)
all_neurons = range(num_neurons_per_layer)
# 補集合の生成
complement_list = get_complement(all_layers, all_neurons, layer_neuron_list)

# # 重複がないかテスト
# # print(has_overlap(layer_neuron_list, complement_list)) # False
# # sys.exit()

""" (activate)したニューロンの中から、layer_neuron_listの補集合を作成: <- つまり、L1 or L2に発火したニューロン """
# ちゃんと降順にとれている確認用に、(layer_idx, neuron_idx, sum_of_act_values)を表示
# layer_neuron_list_L1_or_L2 = []
# for layer_idx, neurons in act_sum_L1_or_L2.items():
#     for neuron_idx, act_sum in neurons.items():
#         layer_neuron_list_L1_or_L2.append((layer_idx, neuron_idx, act_sum))
# layer_neuron_list_L1_or_L2 = sorted(layer_neuron_list_L1_or_L2, key=lambda x: x[2], reverse=True)
# print(layer_neuron_list_L1_or_L2[:10])

layer_neuron_list_L1_or_L2 = []
for layer_idx, neurons in act_sum_L1_or_L2.items():
    for neuron_idx in neurons.keys():
        layer_neuron_list_L1_or_L2.append((layer_idx, neuron_idx))
layer_neuron_list_L1_or_L2 = sorted(layer_neuron_list_L1_or_L2, key=lambda x: act_sum_L1_or_L2[x[0]][x[1]], reverse=True)

""" L1のみに発火しているニューロンの中から、layer_neuron_listの補集合を作成 """
layer_neuron_list_L1_specific = []
for layer_idx, neurons in act_sum_L1_specific.items():
    for neuron_idx in neurons.keys():
        layer_neuron_list_L1_specific.append((layer_idx, neuron_idx))
layer_neuron_list_L1_specific = sorted(layer_neuron_list_L1_specific, key=lambda x: act_sum_L1_specific[x[0]][x[1]], reverse=True)

""" どのくらい介入するか(n) """
intervention_num = c_after
layer_neuron_list = layer_neuron_list[:intervention_num]
complement_list = complement_list[:intervention_num]
layer_neuron_list_L1_or_L2 = layer_neuron_list_L1_or_L2[:intervention_num]
layer_neuron_list_L1_specific = layer_neuron_list_L1_specific[:intervention_num]

"""
上で作成したリストから、指定した範囲の（特定の）layer_idxのみを保持するリストを作成
（特定の層の影響を調べるため。
"""
# # layer range
# layer_range = range(10, 21)  # 10~20

# # 範囲内のlayer_idxに対応するサブリストを作成
# layer_neuron_list = [pair for pair in layer_neuron_list if pair[0] in layer_range]
# complement_list = [pair for pair in complement_list if pair[0] in layer_range]
# layer_neuron_list_L1_or_L2 = [pair for pair in layer_neuron_list_L1_or_L2 if pair[0] in layer_range]
# layer_neuron_list_L1_specific = [pair for pair in layer_neuron_list_L1_specific if pair[0] in layer_range]

# layer_neuron_list = layer_neuron_list[:intervention_num]
# complement_list = complement_list[:intervention_num]
# layer_neuron_list_L1_or_L2 = layer_neuron_list_L1_or_L2[:intervention_num]
# layer_neuron_list_L1_specific = layer_neuron_list_L1_specific[:intervention_num]

# print(layer_neuron_list)
# print(complement_list)
# print(layer_neuron_list_L1_or_L2)
# print(layer_neuron_list_L1_specific)
# sys.exit()


""" neuron intervention (発火値の改竄実験)"""

""" models """
# LLaMA-3
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B"
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    # "nl": "ReBatch/Llama-3-8B-dutch", # du
    # "it": "DeepMount00/Llama-3-8b-Ita", # ita
    # "ko": "beomi/Llama-3-KoEn-8B", # ko
}

# load BLiMP
# BLiMPの評価項目リスト
configs = get_dataset_config_names("blimp")
# configs = ["npi_present_1"]
# configs = ["existential_there_quantifiers_2", "matrix_question_npi_licensor_present"]
# configs = ["existential_there_quantifiers_2"]
# configs = configs[:2]
# sys.exit()
# データを保存するリスト

def eval_blimp(model_names, layer_neuron_list):
    results = []
    # 各モデルについてBLiMPのタスクを評価
    for L2, model_name in model_names.items():
        # load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 各評価項目ごとに評価
        for config in configs:
            blimp = load_dataset("blimp", config)
            correct = 0
            total = 0
            for example in blimp["train"]:
                sentence1 = example["sentence_good"]
                sentence2 = example["sentence_bad"]
                score1, score2 = evaluate_sentence_pair_with_edit_activation(model, tokenizer, layer_neuron_list, sentence1, sentence2)

                if score1 > score2:
                    correct += 1
                total += 1

            # 精度を計算して結果を保存
            accuracy = correct / total
            results.append({
                "Model": model_name,
                "Task": config,
                "Accuracy": accuracy
            })
    return results

if __name__ == "__main__":
    result_main = eval_blimp(model_names, layer_neuron_list)
    print(f"result_main: {result_main}")
    result_comp = eval_blimp(model_names, complement_list)
    print(f"result_comp: {result_comp}")
    result_comp_L1_or_L2 = eval_blimp(model_names, layer_neuron_list_L1_or_L2)
    print(f"result_comp_L1_or_L2: {result_comp_L1_or_L2}")
    result_comp_L1_specific = eval_blimp(model_names, layer_neuron_list_L1_specific)
    print(f"result_comp_L1_specific: {result_comp_L1_specific}")

    # sys.exit()

    # データフレームに変換
    df_main = pd.DataFrame(result_main)
    print(df_main)
    df_comp = pd.DataFrame(result_comp)
    print(df_comp)
    df_comp_L1_or_L2 = pd.DataFrame(result_comp_L1_or_L2)
    print(df_comp_L1_or_L2)
    df_comp_L1_specific = pd.DataFrame(result_comp_L1_specific)
    print(df_comp_L1_specific)

    # 各モデルごとに正答率の平均を計算
    overall_accuracy_main = df_main.groupby('Model')['Accuracy'].mean().reset_index()
    print(overall_accuracy_main)
    overall_accuracy_comp = df_comp.groupby('Model')['Accuracy'].mean().reset_index()
    print(overall_accuracy_comp)
    overall_accuracy_comp_L1_or_L2 = df_comp_L1_or_L2.groupby('Model')['Accuracy'].mean().reset_index()
    print(overall_accuracy_comp_L1_or_L2)
    overall_accuracy_comp_L1_specific = df_comp_L1_specific.groupby('Model')['Accuracy'].mean().reset_index()
    print(overall_accuracy_comp_L1_specific)

    # 列名を変更してOVERALLにします
    overall_accuracy_main.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    overall_accuracy_comp.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    overall_accuracy_comp_L1_or_L2.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    overall_accuracy_comp_L1_specific.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)

    """ CSVに保存 """
    """ all layers """
    # shared_neurons intervention
    df_main.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/shared/n_{intervention_num}/llama3_en_ja_shared_ONLY.csv", index=False)
    # COMPLEMENT of shared_neurons intervention
    df_comp.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/normal_COMP/n_{intervention_num}/llama3_en_ja_COMP.csv", index=False)
    # act_L1_or_L2 intervention
    df_comp_L1_or_L2.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/L1_or_L2/n_{intervention_num}/llama3_en_ja_L1_or_L2.csv", index=False)
    # L1_specific intervention
    df_comp_L1_specific.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/L1_specific/n_{intervention_num}/llama3_en_ja_L1_specific.csv", index=False)

    """ mid layers """
    # # shared_neurons intervention
    # df_main.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/shared/midlayers/n_{intervention_num}/llama3_en_ja_shared.csv", index=False)
    # # COMPLEMENT of shared_neurons intervention
    # df_comp.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/normal_COMP/midlayers/n_{intervention_num}/llama3_en_ja_COMP.csv", index=False)
    # # act_L1_or_L2 intervention
    # df_comp_L1_or_L2.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/L1_or_L2/midlayers/n_{intervention_num}/llama3_ja_ko_L1_or_L2.csv", index=False)
    # # L1_specific intervention
    # df_comp_L1_specific.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/L1_specific/midlayers/n_{intervention_num}/llama3_en_ja_L1_specific.csv", index=False)

    print(f"intervention num: {intervention_num}")
    print(f"THRESHOLD: {THRESHOLD}")
    # print(f"layer_range: {layer_range}")
    print("completed. saved to csv.")
