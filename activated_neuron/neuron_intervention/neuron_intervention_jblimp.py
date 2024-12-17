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
# en_ja
pkl_file_path = "/home/s2410121/proj_LA/activated_neuron/pickles/act_sum/tatoeba_0_th/act_sum_dict/act_sum_dict_en_ja_tatoeba_0_th.pkl"
with open(pkl_file_path, "rb") as f:
    act_sum_dict = pickle.load(f)
print("unfolded pickle")

# それぞれのneuronsの発火値の合計（dict)を取得
act_sum_shared = act_sum_dict["shared"]
act_sum_L1_or_L2 = act_sum_dict["L1_or_L2"]
act_sum_L2_specific = act_sum_dict["L1_specific"]
act_sum_L2_specific = act_sum_dict["L2_specific"]

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
layer_neuron_list = []
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
# ちゃんと降順にとれている確認用に、(layer_idx, neuron_idx, sum_of_act_values)を表示
# layer_neuron_list_L2_specific = []
# for layer_idx, neurons in act_sum_L2_specific.items():
#     for neuron_idx, act_sum in neurons.items():
#         layer_neuron_list_L1_specific.append((layer_idx, neuron_idx, act_sum))
# layer_neuron_list_L2_specific = sorted(layer_neuron_list_L2_specific, key=lambda x: x[2], reverse=True)
# print(layer_neuron_list_L1_specific[:10])

layer_neuron_list_L2_specific = []
for layer_idx, neurons in act_sum_L2_specific.items():
    for neuron_idx in neurons.keys():
        layer_neuron_list_L2_specific.append((layer_idx, neuron_idx))
layer_neuron_list_L2_specific = sorted(layer_neuron_list_L2_specific, key=lambda x: act_sum_L2_specific[x[0]][x[1]], reverse=True)
# print(layer_neuron_list_L1_specific[:10])
# sys.exit()

# どのくらい介入するか(n)
intervention_num = 3000
layer_neuron_list = layer_neuron_list[:intervention_num]
complement_list = complement_list[:intervention_num]
# L1かL2の発火している、layer_neuron_listの補集合
layer_neuron_list_L1_or_L2 = layer_neuron_list_L1_or_L2[:intervention_num]
# L1_specific
layer_neuron_list_L2_specific = layer_neuron_list_L2_specific[:intervention_num]
# print(layer_neuron_list[:10])
# print(complement_list[:10])
# sys.exit()
"""
上で作成したリストから、指定した範囲の（特定の）layer_idxのみを保持するリストを作成
（特定の層の影響を調べるため。
"""
# layer range
# layer_range = range(10, 16)  # 10〜15　

# 範囲内のlayer_idxに対応するサブリストを作成
# sublist_main = [pair for pair in layer_neuron_list if pair[0] in layer_range]
# sublist_comp = [pair for pair in complement_list if pair[0] in layer_range]
# print(sublist_main)
# print(len(sublist_main))
# print(sublist_comp[:len(sublist_main)])
# print(len(sublist_comp[:len(sublist_main)]))
# sys.exit()

""" neuron intervention (発火値の改竄実験)"""

""" models """
# LLaMA-3
model_name = "tokyotech-llm/Llama-3-Swallow-8B-v0.1"
# JBLiMP
jblimp = load_dataset("polm-stability/jblimp")
device = "cuda" if torch.cuda.is_available() else "cpu"

def eval_jblimp(model_names, layer_neuron_list):
    results = []
    predictions = defaultdict(lambda: defaultdict(int))
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for example in jblimp["train"]:
        task_name = example["phenomenon"]
        sentence1 = example["good_sentence"]
        sentence2 = example["bad_sentence"]
        score1, score2 = evaluate_sentence_pair_with_edit_activation(model, tokenizer, layer_neuron_list, sentence1, sentence2)
        if score1 > score2:
            predictions[task_name]["correct"] += 1
        predictions[task_name]["total"] += 1

    # 精度を計算して結果を保存
    for task_name, scores in predictions.items():
        results.append({
            "Model": model_name,
            "Task": task_name,
            "Accuracy": scores["correct"] / scores["total"]
        })

    return results

if __name__ == "__main__":
    results = []
    """
    predictions:
    {
    "task_name":
            "total": num_of_total_sentence(int)
            "correct": num_of_correct_predictions(int)
    }
    """
    result_main = eval_jblimp(model_name, layer_neuron_list)
    print(f"result_main: {result_main}")
    result_comp = eval_jblimp(model_name, complement_list)
    print(f"result_comp: {result_comp}")
    result_comp_L1_or_L2 = eval_jblimp(model_name, layer_neuron_list_L1_or_L2)
    print(f"result_comp_L1_or_L2: {result_comp_L1_or_L2}")
    result_comp_L2_specific = eval_jblimp(model_name, layer_neuron_list_L2_specific)
    print(f"result_comp_L1_specific: {result_comp_L2_specific}")

    # sys.exit()

    # データフレームに変換
    df_main = pd.DataFrame(result_main)
    print(df_main)
    df_comp = pd.DataFrame(result_comp)
    print(df_comp)
    df_comp_L1_or_L2 = pd.DataFrame(result_comp_L1_or_L2)
    print(df_comp_L1_or_L2)
    df_comp_L2_specific = pd.DataFrame(result_comp_L2_specific)
    print(df_comp_L2_specific)

    # 各モデルごとに正答率の平均を計算
    overall_accuracy_main = df_main.groupby('Model')['Accuracy'].mean().reset_index()
    print(overall_accuracy_main)
    overall_accuracy_comp = df_comp.groupby('Model')['Accuracy'].mean().reset_index()
    print(overall_accuracy_comp)
    overall_accuracy_comp_L1_or_L2 = df_comp_L1_or_L2.groupby('Model')['Accuracy'].mean().reset_index()
    print(overall_accuracy_comp_L1_or_L2)
    overall_accuracy_comp_L2_specific = df_comp_L2_specific.groupby('Model')['Accuracy'].mean().reset_index()
    print(overall_accuracy_comp_L2_specific)

    # 列名を変更してOVERALLにします
    overall_accuracy_main.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    overall_accuracy_comp.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    overall_accuracy_comp_L1_or_L2.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
    overall_accuracy_comp_L2_specific.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)

    """ CSVに保存 """
    # shared_neurons intervention
    df_main.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/jblimp/shared/n_{intervention_num}/llama3_en_ja.csv", index=False)
    # COMPLEMENT of shared_neurons intervention
    df_comp.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/jblimp/normal_COMP/n_{intervention_num}/llama3_en_ja_COMP.csv", index=False)
    # act_L1_or_L2 intervention
    df_comp_L1_or_L2.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/jblimp/L1_or_L2/n_{intervention_num}/llama3_en_ja_L1_or_L2.csv", index=False)
    # L1_specific intervention
    df_comp_L2_specific.to_csv(f"/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/jblimp/L2_specific/n_{intervention_num}/llama3_en_ja_L2_specific.csv", index=False)

    print("completed. saved to csv.")
