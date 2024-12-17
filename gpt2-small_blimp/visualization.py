"""
各L2モデルごとに、どれくらいbaseモデルとBLiMPの各項目において差があるかを可視化
L2の追学習によって各項目の精度は上がっているのか、下がっているのか....
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

""" BLiMPの全67項目ごと """
""" LLaMA-3 """

# CSVファイルの読み込み
df = pd.read_csv('/home/s2410121/proj_LA/gpt2-small_blimp/csv_files_final/blimp_evaluation_results_complete2_llama3_all_final.csv')

# LLaMA-3のモデル名
llama_model_names = {
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}

# ベースモデル（meta-llama/Meta-Llama-3-8B）のデータを抽出
base_model = df[df['Model'] == 'meta-llama/Meta-Llama-3-8B']

# 他のモデルのデータを抽出
other_models = df[df['Model'] != 'meta-llama/Meta-Llama-3-8B']

# ベースモデルと他のモデルの精度差を計算
# 他のモデルとベースモデルの精度をタスクごとに比較
comparison_df = other_models.merge(base_model[['Task', 'Accuracy']], on='Task', suffixes=('_other', '_base'))

# 精度差を計算
comparison_df['Accuracy_diff'] = comparison_df['Accuracy_other'] - comparison_df['Accuracy_base']

# フルモデル名をキー（例: "ja"）に変換するマッピングを作成
reverse_llama_model_names = {v: k for k, v in llama_model_names.items()}

# モデルごとにグラフを作成して保存
output_dir = '/home/s2410121/proj_LA/gpt2-small_blimp/images/llama3/acc_comparison_final/'

for full_model_name in comparison_df['Model'].unique():
    # キーを取得（"ja", "nl", etc.）
    model_key = reverse_llama_model_names.get(full_model_name, "unknown_model")

    # 各モデルのデータを抽出
    model_comparison = comparison_df[comparison_df['Model'] == full_model_name]

    # 可視化
    plt.figure(figsize=(12, 8))
    sns.barplot(data=model_comparison, x='Task', y='Accuracy_diff', color='orange')

    # グラフの設定
    plt.title(f'Accuracy Difference for Base Model vs {model_key}')
    plt.xlabel('Task')
    plt.ylabel('Accuracy Difference')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # グラフを保存
    filename = f"{output_dir}{model_key}_accuracy_comparison.png"
    plt.savefig(filename)
    plt.close()

    print(f"Saved: {filename}")


""" gpt2 """
# # CSVファイルの読み込み
# df = pd.read_csv('/home/s2410121/proj_LA/gpt2-small_blimp/csv_files_final/blimp_evaluation_results_complete2_gpt2_all_final.csv')

# # GPT-2-small
# model_names = {
#     "ja": "rinna/japanese-gpt2-small",  # ja
#     "nl": "GroNLP/gpt2-small-dutch",    # du
#     "it": "GroNLP/gpt2-small-italian",  # ita
#     "fr": "dbddv01/gpt2-french-small",  # fre
#     "ko": "skt/kogpt2-base-v2",         # ko
#     "es": "datificate/gpt2-small-spanish"  # spa
# }

# # ベースモデル（gpt2）のデータを抽出
# base_model = df[df['Model'] == 'gpt2']

# # 他のモデルのデータを抽出
# other_models = df[df['Model'] != 'gpt2']

# # ベースモデルと他のモデルの精度差を計算
# comparison_df = other_models.merge(
#     base_model[['Task', 'Accuracy']], on='Task', suffixes=('_other', '_base')
# )
# comparison_df['Accuracy_diff'] = comparison_df['Accuracy_other'] - comparison_df['Accuracy_base']

# # フルモデル名をキー（例: "ja"）に変換するマッピングを作成
# reverse_model_names = {v: k for k, v in model_names.items()}

# # モデルごとにグラフを作成して保存
# output_dir = '/home/s2410121/proj_LA/gpt2-small_blimp/images/gpt2/acc_comparison_final'
# for full_model_name in comparison_df['Model'].unique():
#     # キーを取得（"ja", "nl", etc.）
#     model_key = reverse_model_names.get(full_model_name, "unknown_model")

#     # 各モデルのデータを抽出
#     model_comparison = comparison_df[comparison_df['Model'] == full_model_name]

#     # 可視化
#     plt.figure(figsize=(12, 8))
#     sns.barplot(data=model_comparison, x='Task', y='Accuracy_diff', color='orange')

#     # グラフの設定
#     plt.title(f'Accuracy Difference for Base Model vs {model_key}')
#     plt.xlabel('Task')
#     plt.ylabel('Accuracy Difference')
#     plt.xticks(rotation=90)
#     plt.tight_layout()

#     # グラフを保存
#     filename = f"{output_dir}{model_key}_accuracy_comparison.png"
#     plt.savefig(filename)
#     plt.close()

#     print(f"Saved: {filename}")
