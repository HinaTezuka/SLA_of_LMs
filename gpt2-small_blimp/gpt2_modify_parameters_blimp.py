import os
import sys
sys.path.append('/home/s2410121/proj_LA/measure_similarities')
sys.path.append('/home/s2410121/proj_LA/gpt2-small_blimp')

import torch
import pandas as pd

from collections import defaultdict
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
# import gpt2_eval_blimp
from similarity_funcs import *

# モデルの指定
gpt2_model_original_name = "gpt2" # original gpt2(small) model : en
gpt2_model_japanese_name = "rinna/japanese-gpt2-small" # ja
gpt2_model_dutch_name = "GroNLP/gpt2-small-dutch" # du
gpt2_model_german_name = "dbmdz/german-gpt2" # ger
# gpt2_model_german_name = "ml6team/gpt2-small-german-finetune-oscar" # ger
gpt2_model_italian_name = "GroNLP/gpt2-small-italian" # ita
gpt2_model_french_name = "dbddv01/gpt2-french-small" # fre
gpt2_model_korean_name = "skt/kogpt2-base-v2" # ko
gpt2_model_spanish_name = "datificate/gpt2-small-spanish" # spa

# モデルのロード
gpt2_model_original = AutoModelForCausalLM.from_pretrained(gpt2_model_original_name) # en
gpt2_model_ja = AutoModelForCausalLM.from_pretrained(gpt2_model_japanese_name) # ja
gpt2_model_du = AutoModelForCausalLM.from_pretrained(gpt2_model_dutch_name) # du
gpt2_model_ger = AutoModelForCausalLM.from_pretrained(gpt2_model_german_name) # ger
gpt2_model_ita = AutoModelForCausalLM.from_pretrained(gpt2_model_italian_name) # ita
gpt2_model_fre = AutoModelForCausalLM.from_pretrained(gpt2_model_french_name) # fre
gpt2_model_ko = AutoModelForCausalLM.from_pretrained(gpt2_model_korean_name) # ko
gpt2_model_spa = AutoModelForCausalLM.from_pretrained(gpt2_model_spanish_name) # spa

""" それぞれのモデルのparametersを取得 """

# parametersを取得
state_dict_gpt2_original = gpt2_model_original.state_dict() # original(english)
state_dict_gpt2_ja = gpt2_model_ja.state_dict() # japanese
state_dict_gpt2_du = gpt2_model_du.state_dict() # dutch
state_dict_gpt2_ger = gpt2_model_ger.state_dict() # german
state_dict_gpt2_ita = gpt2_model_ita.state_dict() # italian
state_dict_gpt2_fre = gpt2_model_fre.state_dict() # french
state_dict_gpt2_ko = gpt2_model_ko.state_dict() # ko
state_dict_gpt2_spa = gpt2_model_spa.state_dict() # spanish
# print(state_dict_gpt2_du)

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

def evaluate_sentence_pair(model, tokenizer, sentence1, sentence2):
    inputs1 = tokenizer(sentence1, return_tensors="pt")
    inputs2 = tokenizer(sentence2, return_tensors="pt")

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    """ モデルがそれぞれの文を生成する確率 """
    # score1 = outputs1.logits.log_softmax(dim=-1)[..., inputs1.input_ids[0]].sum()
    # score2 = outputs2.logits.log_softmax(dim=-1)[..., inputs2.input_ids[0]].sum()

    # 文1の対数確率をトークンごとに取得して平均
    log_probs1 = outputs1.logits.log_softmax(dim=-1)
    score1 = log_probs1[..., inputs1.input_ids[0]].mean()

    # 文2の対数確率をトークンごとに取得して平均
    log_probs2 = outputs2.logits.log_softmax(dim=-1)
    score2 = log_probs2[..., inputs2.input_ids[0]].mean()

    return score1, score2


""" それぞれのL2モデルのmodified versionを作成 / すべてのモデルを用意 """
models = []
model_names = [
                "rinna/japanese-gpt2-small", # ja
                "GroNLP/gpt2-small-dutch", # du
                "dbmdz/german-gpt2", # ger
                "GroNLP/gpt2-small-italian", # ita
                "dbddv01/gpt2-french-small", # fre
                "skt/kogpt2-base-v2", # ko
                "datificate/gpt2-small-spanish" # spa
            ]
topk = 6

""" japanese """
# topkこのoutliers(base modelと比べて、一番差があるパラメータを持つ層)を取得
topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ja, topk)
# print(topk_outliers.values())
# sys.exit()
# topkこの層のパラメータに対して緩和操作を実施 <- 戻り値はstate_dict
modified_ja_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ja, topk_outliers)
# L1->L2モデルのロード
gpt2_model_ja.load_state_dict(modified_ja_dict)
# modelsに追加
models.append(gpt2_model_ja)

""" dutch """
topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_du, topk)
modified_du_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_du, topk_outliers)
gpt2_model_du.load_state_dict(modified_du_dict)
models.append(gpt2_model_du)

""" german """
topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ger, topk)
modified_ger_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ger, topk_outliers)
gpt2_model_ger.load_state_dict(modified_ger_dict)
models.append(gpt2_model_ger)

""" italy """
topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ita, topk)
modified_ger_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ita, topk_outliers)
gpt2_model_ita.load_state_dict(modified_ita_dict)
models.append(gpt2_model_ita)

""" french """
topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_fre, topk)
modified_fre_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_fre, topk_outliers)
gpt2_model_fre.load_state_dict(modified_fre_dict)
models.append(gpt2_model_fre)

""" korean """
topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ko, topk)
modified_ko_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ko, topk_outliers)
gpt2_model_ko.load_state_dict(modified_ko_dict)
models.append(gpt2_model_ko)

""" spanish """
topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_spa, topk)
modified_spa_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_spa, topk_outliers)
gpt2_model_spa.load_state_dict(modified_spa_dict)
models.append(gpt2_model_spa)

""" それぞれ、modify前のモデルと、modify後のBLiMPの精度を測る """
# BLiMPの評価項目リスト
configs = get_dataset_config_names("blimp")

results = []
# 各モデルについてBLiMPのタスクを評価
for model, model_name in zip(models, model_names):
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
            score1, score2 = evaluate_sentence_pair(model, tokenizer, sentence1, sentence2)

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

# 各モデルごとに正答率の平均を計算
overall_accuracy = df.groupby('Model')['Accuracy'].mean().reset_index()
print(f'overall accuracy: {overall_accuracy}')

# 列名を変更してOVERALLにします
overall_accuracy.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)

print(df)
# CSVに保存
df.to_csv("blimp_gpt2_modified_ALL_1_2.csv", index=False)

print("評価結果をcsv fileに保存しました。")
