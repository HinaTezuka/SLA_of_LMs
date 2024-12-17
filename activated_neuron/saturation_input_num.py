""" どの程度input文をいれたら検出ニューロン数がサチるのか検証 """

import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from baukit import Trace, TraceDict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel

from neuron_detection_funcs import track_neurons_with_text_data

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

L1 = "en" # L1 is fixed to english.

""" 検証する文数のpattern """
# num_input_patterns = [100, 500, 1000, 2000, 3000, 10000, 20000]
# num_input_patterns = [1, 2]
""" test """
rows_to_add = []

for L2, model_name in model_names.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    # データセットを取得し、指定の文数だけ選択
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    dataset = dataset.select(range(100))
    tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]

    # ニューロンの検出
    neuron_detection_dict = track_neurons_with_text_data(model, tokenizer, tatoeba_data, 0, 0)

    num_layers = 32
    for layer_idx in range(num_layers):
        # 各層のニューロン集合を取得
        L1_neurons_set = set(neuron_detection_dict["activated_neurons_L1"][layer_idx][1][:, 2])
        L2_neurons_set = set(neuron_detection_dict["activated_neurons_L2"][layer_idx][1][:, 2])
        shared_neurons_set = set(neuron_detection_dict["shared_neurons"][layer_idx][1].flatten())
        specific_L1_neurons_set = set(neuron_detection_dict["specific_neurons_L1"][layer_idx][1].flatten())
        specific_L2_neurons_set = set(neuron_detection_dict["specific_neurons_L2"][layer_idx][1].flatten())

        # 追加する行をリストとして保存
        rows_to_add.append({
            "L2 Language": L2,
            "Num Inputs": num,
            "Layer": layer_idx + 1,
            "Activated Neurons L1": len(L1_neurons_set),
            "Activated Neurons L2": len(L2_neurons_set),
            "Shared Neurons": len(shared_neurons_set),
            "Specific Neurons L1": len(specific_L1_neurons_set),
            "Specific Neurons L2": len(specific_L2_neurons_set)
        })
    # GPUメモリの解放
    torch.cuda.empty_cache()

# pd.concatを使ってDataFrameに追加
layer_results_df = pd.concat([layer_results_df, pd.DataFrame(rows_to_add)], ignore_index=True)

# データフレームを画像として保存する関数
def save_df_as_image(df, file_path):
    fig, ax = plt.subplots(figsize=(15, len(df) * 0.3))  # テーブルサイズ調整
    ax.axis('tight')
    ax.axis('off')

    # データフレームをテーブル形式で表示
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.5, 1.5)  # スケール調整

    # 画像として保存
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()

# レイヤーごとのデータフレームを画像として保存
save_df_as_image(layer_results_df, "layerwise_neuron_counts_summary.png")
