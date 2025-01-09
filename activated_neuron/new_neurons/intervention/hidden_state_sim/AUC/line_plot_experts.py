"""
visualize experts for same semantics as a sentence.
"""
import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/hidden_state_sim/AUC")
import dill as pickle
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from expertise_funcs import (
    save_as_pickle,
    unfreeze_pickle,
)


""" parameters setting """
activation_type = "abs"
# activation_type = "product"
norm_type = "no"
# norm_type = "min_max"
# norm_type = "sigmoid"
# L2 = "ja"
L2_list = ["ja", "nl", "ko", "it"]

# データを格納するリスト
data = []

for L2 in L2_list:
    # LLaMA-3 unfreezing
    sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/sorted_neurons_{L2}.pkl"
    llama3_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/ap_scores_{L2}.pkl"
    sorted_neurons_llama3 = unfreeze_pickle(sorted_neurons_path)
    ap_scores_llama3 = unfreeze_pickle(llama3_path)

    # GPT-2 unfreezing
    sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/sorted_neurons_{L2}.pkl"
    gpt2_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/ap_scores_{L2}.pkl"
    sorted_neurons_gpt2 = unfreeze_pickle(sorted_neurons_path)
    ap_scores_gpt2 = unfreeze_pickle(gpt2_path)

    # makind dataframe for visualization
    for i in range(len(sorted_neurons_llama3[:30000])):
        data.append({"Model": "LLaMA-3", "Language": L2, "Neuron": i, "AP_Score": ap_scores_llama3[sorted_neurons_llama3[i]]})
    for i in range(len(sorted_neurons_gpt2[:30000])):
        data.append({"Model": "GPT-2", "Language": L2, "Neuron": i, "AP_Score": ap_scores_gpt2[sorted_neurons_gpt2[i]]})

# convert pandas DataFrame
df = pd.DataFrame(data)

# AP_Score を数値に変換
df['AP_Score'] = pd.to_numeric(df['AP_Score'], errors='coerce')

# Neuron カラムを数値に変換
df['Neuron'] = pd.to_numeric(df['Neuron'], errors='coerce')

# Language と Model を文字列型に変換
df['Language'] = df['Language'].astype(str)
df['Model'] = df['Model'].astype(str)

# プロット作成
plt.figure(figsize=(16, 10))  # グラフサイズを大きめに調整
sns.lineplot(
    data=df,
    x="Neuron",
    y="AP_Score",
    hue=f"Language",
    style="Model",
    markers=False,
    # dashes=False,
    dashes={"GPT-2": [2, 2], "LLaMA-3": [1, 0]},
    palette=sns.color_palette("tab10"),  # カラーパレット設定
    linewidth=5,
    alpha=1,
)

# グラフの装飾
plt.xlabel("Neuron Index", fontsize=30)
plt.ylabel("AP Score", fontsize=30)
plt.title("AP Scores for LLaMA-3 and GPT-2", fontsize=25)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.legend(title=f"Language & Model", fontsize=20, title_fontsize=20)
plt.grid(True, linestyle="--", alpha=0.6)

# 保存と表示
plt.tight_layout()
plt.savefig(
    f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/same_semantic_experts/{activation_type}/ap_scores_comparison.png",
    bbox_inches="tight"
)
