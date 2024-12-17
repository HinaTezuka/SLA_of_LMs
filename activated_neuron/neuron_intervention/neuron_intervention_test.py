"""
ちゃんと発火値が改竄できているかtest
"""

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


""" Test """

""" 複数の (layer_idx, neuron_idx) のペアをリストに定義 """

""" randomリストを作成(test用) """
import random
# リストの長さ
list_length = 4000
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
layer_idx_and_neuron_idx.append((0, 0))
layer_idx_and_neuron_idx.append((0, 1))
layer_idx_and_neuron_idx.append((0, 14335))
layer_idx_and_neuron_idx.append((1, 0))
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

# トレースされた出力を格納する辞書(ちゃんと改竄が反映されているか)
traced_activations = {}

# Traceで複数のレイヤーを追跡
trace_layers = [f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_idx_and_neuron_idx]
with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_idx_and_neuron_idx)) as tr:
    # モデルの入力を設定
    input_t = 'こんにちは。今日は'
    input_ids = tokenizer(input_t, return_tensors="pt")
    # モデル推論
    output = model.generate(input_ids["input_ids"])

    # logits ver.
    with torch.no_grad():
        output_logits = model(**input_ids)
    print(output_logits.logits.log_softmax(dim=-1))
    # sys.exit()

    # 改竄後の各層の発火値を記録(ちゃんと改竄できているか)
    for layer_name, output in tr.items():
        traced_activations[layer_name] = output.output

    # print(tokenizer.decode(output[0], skip_special_tokens=True))

# TraceDictが終了した後、通常の推論を再確認
with torch.no_grad():
    output_normal = model(**input_ids)
print(output_normal.logits.log_softmax(dim=-1))

with torch.no_grad():
    output_normal = model(**input_ids)
print(output_normal.logits.log_softmax(dim=-1))

# トレースされた発火値を出力
for layer_name, activation in traced_activations.items():
    print(f"Layer: {layer_name}, Activation shape: {activation}")
    print(f"Layer: {layer_name}, Activation shape: {activation.shape}")
