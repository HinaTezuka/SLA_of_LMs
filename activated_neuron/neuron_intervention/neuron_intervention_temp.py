import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")

import numpy as np
import matplotlib.pyplot as plt
import torch
import baukit
from baukit import Trace, TraceDict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel

# from neuron_detection_funcs import neuron_detection_dict


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
model = AutoModelForCausalLM.from_pretrained(model_names["ja"])
tokenizer = AutoTokenizer.from_pretrained(model_names["ja"])
# Baukit
# def edit_activation(output, layer):
#     # ここでoutputを編集する
#     return output * 0.01  # 活性化値を0にする

# input_t = '太郎さん、こんに'
# with Trace(model, 'model.layers.31.mlp.act_fn', edit_output=edit_activation) as tr:
#     input_ids = tokenizer(input_t, return_tensors="pt")
#     output = model.generate(input_ids["input_ids"])  # モデル推論
#     print("Edited activation values:", tr.output)  # 編集された活性化値
#     print(tokenizer.decode(output[0], skip_special_tokens=True))

""" """
# import random

# # レイヤー番号31は固定
# layer_idx = 31

# # ニューロンインデックスの1から14000のユニークな数をランダムに選択
# neuron_indices = random.sample(range(1, 14001), 1000)

# # (レイヤー番号, ニューロンインデックス) のタプルを作成
# layer_idx_and_neuron_idx = [(layer_idx, idx) for idx in neuron_indices]

# # 結果を確認
# print(len(layer_idx_and_neuron_idx))  # 1000
# print(layer_idx_and_neuron_idx[:5])  # 最初の5つのタプルを表示
# sys.exit()

# def edit_activation(output, layer, layer_idx_and_neuron_idx):
#     """
#     特定のレイヤーインデックスと複数のニューロンインデックスを指定して活性化値を操作
#     :param output: 活性化値
#     :param layer: 現在のレイヤー情報
#     :param layer_idx_and_neuron_idx: 対象のレイヤーインデックスとニューロンインデックスのリスト
#     """
#     for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
#         # 指定されたレイヤーに対して操作を行う
#         if str(layer_idx) in layer:
#             # ニューロンのインデックスが範囲内かをチェック
#             if neuron_idx < output.shape[2]:
#                 # 指定されたニューロンの活性化値を変更
#                 # for
#                 output[:, :, neuron_idx] *= 0
#             else:
#                 print(f"Warning: neuron_idx {neuron_idx} is out of bounds for output with shape {output.shape}")
#         else:
#             print(f"Warning: layer_idx {layer_idx} is not found in the layer.")

#     return output

# # モデルの入力を設定
# input_t = 'こんにちは。今日は'
# input_ids = tokenizer(input_t, return_tensors="pt")
# layer_idx = 31

# # Traceの開始
# with Trace(model, f'model.layers.{layer_idx}.mlp.act_fn', edit_output=lambda output, layer: edit_activation(output, layer, layer_idx_and_neuron_idx)) as tr:
#     output = model.generate(input_ids["input_ids"])  # モデル推論
#     print("Edited activation values:", tr.output)  # 編集された活性化値
#     print(tokenizer.decode(output[0], skip_special_tokens=True))

""" Test """

""" 複数の (layer_idx, neuron_idx) のペアをリストに定義 """

""" randomリストを作成(test用) """
import random
# リストの長さ
list_length = 2000
# レイヤーインデックスの範囲
layer_indices = list(range(32))  # 0から31までのレイヤーインデックス

# ニューロンインデックスの範囲
neuron_index_range = (0, 14335)
# 最終リストを作成
layer_idx_and_neuron_idx = []
# 各layer_idxに対して、順番に2000個のタプルを生成
for layer_idx in layer_indices:
    # 各layer_idxに対してランダムなneuron_idxをリストに追加
    layer_idx_and_neuron_idx.extend([(layer_idx, random.randint(*neuron_index_range)) for _ in range(list_length // len(layer_indices))])
# 作成したリストを表示
print(layer_idx_and_neuron_idx)
print(len(layer_idx_and_neuron_idx))
# sys.exit()
""" """

""" func for editing activation values """
def edit_activation(output, layer, layer_idx_and_neuron_idx):
    """
    edit activation value of neurons(indexed layer_idx and neuron_idx)
    output: activation values
    layer: sth like 'model.layers.{layer_idx}.mlp.act_fn'
    layer_idx_and_neuron_idx: list of tuples like [(layer_idx, neuron_idx), ....]
    """
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if str(layer_idx) in layer:  # layer名にlayer_idxが含まれているか確認
            if neuron_idx < output.shape[2]:  # ニューロンインデックスが範囲内かチェック
                output[:, :, neuron_idx] *= 0  # 指定されたニューロンの活性化値をゼロに設定
                # print(output[:, :, neuron_idx])
                # print(f"Layer {layer_idx}, Neuron {neuron_idx} activation set to 0")  # 確認用出力
            else:
                print(f"Warning: neuron_idx {neuron_idx} is out of bounds for output with shape {output.shape}")

    return output

# def edit_activation(output, layer, layer_idx_and_neuron_idx):
#     """
#     edit activation value of neurons(indexed layer_idx and neuron_idx)
#     output: activation values
#     layer: sth like 'model.layers.{layer_idx}.mlp.act_fn'
#     layer_idx_and_neuron_idx: list of tuples like [(layer_idx, neuron_idx), ....]
#     """
#     for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
#         if str(layer_idx) in layer and layer_idx != 31:  # layer名にlayer_idxが含まれているか確認
#             if neuron_idx < output.shape[2]:  # ニューロンインデックスが範囲内かチェック
#                 output[:, :, neuron_idx] *= 0  # 指定されたニューロンの活性化値をゼロに設定
#                 # print(output[:, :, neuron_idx])
#                 # print(f"Layer {layer_idx}, Neuron {neuron_idx} activation set to 0")  # 確認用出力
#             else:
#                 print(f"Warning: neuron_idx {neuron_idx} is out of bounds for output with shape {output.shape}")
#         elif str(layer_idx) in layer and layer_idx == 31:
#             if neuron_idx < output.shape[2]:  # ニューロンインデックスが範囲内かチェック
#                 output[:, :, neuron_idx] *= 1

#     return output

# Traceで複数のレイヤーを追跡
trace_layers = [f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_idx_and_neuron_idx]
with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_idx_and_neuron_idx)) as tr:
    # モデルの入力を設定
    input_t = 'こんにちは。今日は'
    input_ids = tokenizer(input_t, return_tensors="pt")
    # モデル推論
    output = model.generate(input_ids["input_ids"])

    # logits ver.
    # with torch.no_grad():
    #     output = model(**input_ids)
    # print(output)
    # sys.exit()

    print(tokenizer.decode(output[0], skip_special_tokens=True))

""" TraceDictではなく、Traceで1つ1つ層ごとに操作 """
# # 特定のレイヤーとニューロンの発火値を編集する関数
# def edit_activation(output, layer_idx_and_neuron_idx, current_layer_idx):
#     """
#     現在のレイヤーで指定されたニューロンの発火値を操作する
#     """
#     for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
#         if layer_idx == current_layer_idx:  # 現在のレイヤーに該当するか
#             if neuron_idx < output.shape[2]:  # ニューロンインデックスが範囲内かチェック
#                 output[:, :, neuron_idx] *= 0  # 発火値をゼロに設定
#             else:
#                 print(f"Warning: neuron_idx {neuron_idx} out of bounds for output shape {output.shape}")
#     return output

# # 発火値を編集するレイヤーとニューロンのリストを生成
# list_length = 14000
# layer_indices = list(range(32))  # 0〜31層
# neuron_index_range = (0, 14335)

# layer_idx_and_neuron_idx = []
# for layer_idx in layer_indices:
#     layer_idx_and_neuron_idx.extend([(layer_idx, random.randint(*neuron_index_range)) for _ in range(list_length // len(layer_indices))])
# print(layer_idx_and_neuron_idx)
# print(len(layer_idx_and_neuron_idx))


# # 編集対象のレイヤーを順番に処理
# input_text = "こんにちは。今日は"
# input_ids = tokenizer(input_text, return_tensors="pt")

# for layer_idx in layer_indices:
#     # レイヤーごとにTraceを設定
#     with Trace(model, f"model.layers.{layer_idx}.mlp.act_fn",
#                edit_output=lambda output, layer: edit_activation(output, layer_idx_and_neuron_idx, layer_idx)) as tr:
#         output = model.generate(input_ids["input_ids"])  # 推論
#         print(f"Layer {layer_idx} processed.")

# # 最終的な出力結果
# print("Generated Text:", tokenizer.decode(output[0], skip_special_tokens=True))
