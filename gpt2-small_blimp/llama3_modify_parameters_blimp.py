import os
import sys
sys.path.append('/home/s2410121/proj_LA/measure_similarities')

import torch

from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
import llama3_eval_blimp
from similarity_funcs import *

# モデルの指定
# モデル指定
llama_model_original_name = "meta-llama/Meta-Llama-3-8B" # en
# llama_model_ja_name = "lightblue/suzume-llama-3-8B-japanese" # ja
llama_model_ja_name = "tokyotech-llm/Llama-3-Swallow-8B-v0.1" # ja
llama_model_ger_name = "DiscoResearch/Llama3-German-8B" # ger
llama_model_ita_name = "DeepMount00/Llama-3-8b-Ita" # ita
# llama_model_ko_name = "beomi/Llama-3-Open-Ko-8B" # korean
llama_model_ko_name = "beomi/Llama-3-KoEn-8B" # korean: enの基盤モデルにen, ko両方のテキストでkoreanをL2として追加学習済み

# モデルのロード
# モデルのロード
llama_model_original = AutoModel.from_pretrained(llama_model_original_name)
llama_model_ja = AutoModel.from_pretrained(llama_model_ja_name)
llama_model_ger = AutoModel.from_pretrained(llama_model_ger_name)
llama_model_ita = AutoModel.from_pretrained(llama_model_ita_name)
llama_model_ko = AutoModel.from_pretrained(llama_model_ko_name)

""" それぞれのモデルのparametersを取得 """

# 重みを取得
state_dict_llama_original = llama_model_original.state_dict() # original(english)
state_dict_llama_ja = llama_model_ja.state_dict() # japanese
state_dict_llama_ger = llama_model_ger.state_dict() # german
state_dict_llama_ita = llama_model_ita.state_dict() # italian
state_dict_llama_ko = llama_model_ko.state_dict() # korean

# 各モデルのstate_dictのkeyの数、keyの名前が等しいことを確認： 確認済み
# print(state_dict_llama_original.keys() == state_dict_llama_ja.keys())
# print(state_dict_llama_original.keys() == state_dict_llama_ja.keys())
# print(state_dict_llama_original.keys() == state_dict_llama_du.keys())
# print(state_dict_llama_original.keys() == state_dict_llama_ger.keys())
# print(state_dict_llama_original.keys() == state_dict_llama_ita.keys())
# print(state_dict_llama_original.keys() == state_dict_llama_fre.keys())
# print(state_dict_llama_original.keys() == state_dict_llama_ko.keys())
# print(state_dict_llama_original.keys() == state_dict_llama_spa.keys())

""" state_dictからoriginl modelと比べて差が大きすぎる特定のparametersを取得(上位kコ)するfunc """
def get_topk_outliers_parameters(m1_dict, m2_dict, topk: int) -> dict:
    """
    絶対値の差が大きすぎるlayerのlayer_nameとparameters(torch.Tensor)をタプルペアとして保存するようのdict(ただしtopk個:更新対象のみ保存:更新対象は L2 modelのparameters)
    key: 絶対値の差(float)
    value: (layer_name: str, m1_dict[layer_name]: torch.Tensor) <- tuple
    """
    abs_diff_descending_dict = {}

    for layer_name in m1_dict.keys():
        # shapeが違う層があったら小さい方のTensorにそろえる
        if m1_dict[layer_name].shape != m2_dict[layer_name].shape:
            m1_dict[layer_name], m2_dict[layer_name] = align_tensor_sizes(m1_dict[layer_name], m2_dict[layer_name])

        # 2つの(modelの)parametersの差を計算(絶対値の差の平均)
        weight_mean_abs_diff = mean_abs_diff(m1_dict[layer_name], m2_dict[layer_name])

        # 辞書に追加
        if len(abs_diff_descending_dict) < topk:
            abs_diff_descending_dict[weight_mean_abs_diff] = (layer_name, m2_dict[layer_name])
        else:
            min_key = min(abs_diff_descending_dict.keys())
            if weight_mean_abs_diff > min_key:
                # 最小要素を削除して新しい要素を追加
                abs_diff_descending_dict.pop(min_key)
                abs_diff_descending_dict[weight_mean_abs_diff] = (layer_name, m2_dict[layer_name])

    # 絶対値の差で降順にソート
    return dict(sorted(abs_diff_descending_dict.items(), reverse=True))

""" topk個の差がある、L2_modelのparametersを緩和するfunc (戻り値はmodifyされたstate_dict)"""
def relax_abs_diff_of_L2_model(original_model_dict, L2_model_dict, topk_abs_diff_dict):
    """
    元の差を0にした新しいL2モデルを返す
    """
    # アウトライヤーのパラメータを緩和
    for abs_diff, (layer_name, l2_param) in topk_abs_diff_dict.items():
        original_param = original_model_dict[layer_name]  # 元のモデルのパラメータ
        # 差を0にして、modified L2モデルのパラメータを更新
        L2_model_dict[layer_name] = original_param

    return L2_model_dict

""" topk個の差がある、L2_modelのparametersを緩和するfunc """
# def relax_abs_diff_of_L2_model(original_model, L2_model, topk_abs_diff_dict):
#     """
#     元の差を1/2程度に緩和した新しいL2モデルを返す
#     """
#     # 新しいモデルを作成
#     modified_L2_model = L2_model.copy()  # L2_modelのコピーを作成

#     # アウトライヤーのパラメータを緩和
#     for abs_diff, (layer_name, l2_param) in topk_abs_diff_dict.items():
#         original_param = original_model[layer_name]  # 元のモデルのパラメータ
#         # 差を1/2程度に緩和する
#         adjusted_param = original_param + ((l2_param - original_param) / 2)
#         # modified L2モデルのパラメータを更新
#         modified_L2_model[layer_name] = adjusted_param

#     return modified_L2_model

""" topk個の差がある、L2_modelのパラメータをないものとするfunc """
# def relax_abs_diff_of_L2_model(original_model, L2_model, topk_abs_diff_dict):
#     """
#     topk個の差がある、L2_modelのパラメータをないものとした新しいL2モデルを返す
#     """
#     # L2_modelのコピーを作成
#     modified_L2_model = L2_model.copy()

#     # outliersのパラメータを緩和
#     for abs_diff, (layer_name, l2_param) in topk_abs_diff_dict.items():
#         original_param = original_model[layer_name]  # 元のモデルのパラメータ取得
#         shape = original_param.shape  # パラメータの形状を取得

#         # 全ての値を1にしたテンソルを作成
#         adjusted_param = torch.ones(shape, dtype=original_param.dtype, device=original_param.device)

#         # modified L2モデルのパラメータを更新
#         modified_L2_model[layer_name] = adjusted_param

#     return modified_L2_model


""" ちゃんと緩和されているかテスト """
# topk = 5
# topk_outliers = get_topk_outliers_parameters(state_dict_llama_original, state_dict_llama_ja, topk)
# modified_L2_model = relax_abs_diff_of_L2_model(state_dict_llama_original, state_dict_llama_ja, topk_outliers)
# print(topk_outliers.keys())
# print('\n')
# print('---------------------------------------------------------------')
# print('\n')
# topk_outliers_modified = get_topk_outliers_parameters(state_dict_llama_original, modified_L2_model, topk)
# print(topk_outliers_modified.keys())

""" それぞれのL2モデルのmodified versionを作成 / すべてのモデルを用意 """
models = []
model_names = [
                llama_model_original_name, # ja
                llama_model_ger_name, # ger
                llama_model_ita_name, # ita
                llama_model_ko_name, # ko
            ]
topk = 10

""" japanese """
# topkこのoutliers(base modelと比べて、一番差があるパラメータを持つ層)を取得
topk_outliers = get_topk_outliers_parameters(state_dict_llama_original, state_dict_llama_ja, topk)
# topkこの層のパラメータに対して緩和操作を実施
modified_ja_model_dict = relax_abs_diff_of_L2_model(state_dict_llama_original, state_dict_llama_ja, topk_outliers)
# L1->L2モデルのロード
llama_model_ja_modified = llama_model_ja.load_state_dict(modified_ja_model_dict)
# modelsに追加
models.append(llama_model_ja_modified)

# """ german """
# topk_outliers = get_topk_outliers_parameters(state_dict_llama_original, state_dict_llama_ger, topk)
# modified_ger_model_dict = relax_abs_diff_of_L2_model(state_dict_llama_original, state_dict_llama_ger, topk_outliers)
# llama_model_ger = AutoModelModel.from_pretrained(llama_model_ger_name)
# llama_model_ger_modified = llama_model_ger.load_state_dict(modified_ger_model_dict)

# # models.append(llama_model_ger)
# models.append(llama_model_ger_modified)

# """ italy """
# topk_outliers = get_topk_outliers_parameters(state_dict_llama_original, state_dict_llama_ita, topk)
# modified_ita_model_dict = relax_abs_diff_of_L2_model(state_dict_llama_original, state_dict_llama_ita, topk_outliers)
# llama_model_ita = AutoModel.from_pretrained(llama_model_ita_name)
# llama_model_ita_modified = llama_model_ger.load_state_dict(modified_ita_model_dict)

# # models.append(llama_model_ger)
# models.append(llama_model_ita_modified)

# """ korean """
# topk_outliers = get_topk_outliers_parameters(state_dict_llama_original, state_dict_llama_ko, topk)
# modified_ko_model_dict = relax_abs_diff_of_L2_model(state_dict_llama_original, state_dict_llama_ko, topk_outliers)
# llama_model_ko = AutoModel.from_pretrained(llama_model_ko_name)
# llama_model_ko_modified = llama_model_ko.load_state_dict(modified_ger_model_dict)

# # models.append(llama_model_ger)
# models.append(llama_model_ko_modified)


""" それぞれ、modify前のモデルと、modify後のBLiMPの精度を測る """
# BLiMPの評価項目リスト
configs = get_dataset_config_names("blimp")

results = []
# 各モデルについてBLiMPのタスクを評価
for model, model_name in zip(models, models_names):
    # トークナイザーをロード
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 各評価項目ごとに評価
    for config in configs:
        blimp = load_dataset("blimp", config)
        correct = 0
        total = 0

        for example in blimp["train"]:
            sentence1 = example["sentence_good"]
            sentence2 = example["sentence_bad"]
            score1, score2 = llama3_eval_blimp.evaluate_sentence_pair(model, tokenizer, sentence1, sentence2)

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


""" result """

# データフレームに変換
df = pd.DataFrame(results)
# print(df)

# 各モデルごとに正答率の平均を計算します
overall_accuracy = df.groupby('Model')['Accuracy'].mean().reset_index()
print(f'overall accuracy: {overall_accuracy}')

# 列名を変更してOVERALLにします
overall_accuracy.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)

print(df)
# CSVに保存
df.to_csv("blimp_llama3_modified_ALL_1_2.csv", index=False)

print("評価結果をcsv fileに保存しました。")
