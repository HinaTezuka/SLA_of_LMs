""" prototype experiment for GPU-node """

import sys
sys.path.append('..')
from llama3_measure_similarity import *
from measure_similarities import similarity_funcs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 重みを取得
state_dict_llama_original = llama_model_original.state_dict() # original(english)
state_dict_llama_ja = llama_model_ja.state_dict() # japanese
state_dict_llama_ger = llama_model_ger.state_dict() # german
state_dict_llama_ita = llama_model_ita.state_dict() # italian
state_dict_llama_ko = llama_model_ko.state_dict() # korean

""" keysの名前が一致してそうか確認 """
# print('--------------- keys lens ------------')
# print(len(state_dict_llama_original) # 290
# print(len(state_dict_llama_ja)) # 290
# print(len(state_dict_llama_ger)) # 290
# print(len(state_dict_llama_ita)) # 290
# print(len(state_dict_llama_ko)) #
# print('------- keys list -----------------')
# print(state_dict_llama_original.keys())
# print(state_dict_llama_ja.keys())
# print(state_dict_llama_ger.keys())
# print(state_dict_llama_ita.keys())
# print(state_dict_llama_ko.keys())
# print('----------- key agreements ----------')
# print(state_dict_llama_original.keys() == state_dict_llama_ja.keys()) # True
# print(state_dict_llama_original.keys() == state_dict_llama_ger.keys()) # True
# print(state_dict_llama_original.keys() == state_dict_llama_ita.keys()) # True
# print(state_dict_llama_original.keys() == state_dict_llama_ko.keys()) #

weight_changes_llama = {} # 全体の結果の保存先

# 個々のモデル・言語ペアの保存先
weight_changes_llama_en_ja = defaultdict(dict)
weight_changes_llama_en_ger = defaultdict(dict)
weight_changes_llama_en_ita = defaultdict(dict)
weight_changes_llama_en_ko = defaultdict(dict)

""" 類似度などを計算 """
""" 英語->日本語 """
# 類似度などを計算
weight_changes_llama_en_ja_computed = similarity_funcs.compare_between_models(weight_changes_llama_en_ja, state_dict_llama_original, state_dict_llama_ja) # en_ja
# 辞書に結果を格納
weight_changes_llama['en_ja'] = weight_changes_llama_en_ja_computed
""" 英語->ドイツ語 """
weight_changes_llama_en_ger_computed = similarity_funcs.compare_between_models(weight_changes_llama_en_ger, state_dict_llama_original, state_dict_llama_ger) # en_du
weight_changes_llama['en_ger'] = weight_changes_llama_en_ger_computed
""" 英語->イタリア語 """
weight_changes_llama_en_ita_computed = similarity_funcs.compare_between_models(weight_changes_llama_en_ita, state_dict_llama_original, state_dict_llama_ita) # en_ita
weight_changes_llama['en_ita'] = weight_changes_llama_en_ita_computed
""" 英語->韓国語 """
weight_changes_llama_en_ko_computed = similarity_funcs.compare_between_models(weight_changes_llama_en_ko, state_dict_llama_original, state_dict_llama_ko) # en_ko
weight_changes_llama['en_ko'] = weight_changes_llama_en_ko_computed


""" 可視化 """
""" bar plot """

# 各モデルの平均絶対差を計算
en_ja_mean_diff = np.mean([values['mean_absolute_difference'] for values in weight_changes_llama['en_ja'].values()])
en_ger_mean_diff = np.mean([values['mean_absolute_difference'] for values in weight_changes_llama['en_ger'].values()])
en_ita_mean_diff = np.mean([values['mean_absolute_difference'] for values in weight_changes_llama['en_ita'].values()])
en_ko_mean_diff = np.mean([values['mean_absolute_difference'] for values in weight_changes_llama['en_ko'].values()])

# 各モデル間のコサイン類似度の平均を計算
en_ja_cos_sim = np.mean([values['cos_sim'] for values in weight_changes_llama['en_ja'].values()])
en_ger_cos_sim = np.mean([values['cos_sim'] for values in weight_changes_llama['en_ger'].values()])
en_ita_cos_sim = np.mean([values['cos_sim'] for values in weight_changes_llama['en_ita'].values()])
en_ko_cos_sim = np.mean([values['cos_sim'] for values in weight_changes_llama['en_ko'].values()])

# 結果をデータフレームにまとめる
mean_diffs = pd.DataFrame({
    'Model': ['en_ja', 'en_ger', 'en_ita', 'en_ko'],
    'Mean Absolute Difference': [en_ja_mean_diff, en_ger_mean_diff, en_ita_mean_diff, en_ko_mean_diff],
    'Cosine Similarity': [en_ja_cos_sim, en_ger_cos_sim, en_ita_cos_sim, en_ko_cos_sim]
})

# 可視化（平均絶対差）
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(mean_diffs['Model'], mean_diffs['Mean Absolute Difference'], color=['blue', 'orange', 'red', 'green'])
plt.xlabel('Model')
plt.ylabel('Mean Absolute Difference')
plt.title('Comparison of Mean Absolute Differences')
plt.ylim(0, max(mean_diffs['Mean Absolute Difference']) * 1.1)  # Y軸の範囲を調整
plt.grid(axis='y')

# 可視化（コサイン類似度）
plt.subplot(1, 2, 2)
plt.bar(mean_diffs['Model'], mean_diffs['Cosine Similarity'], color=['blue', 'orange', 'red', 'green'])
plt.xlabel('Model')
plt.ylabel('Cosine Similarity')
plt.title('Comparison of Cosine Similarity')
plt.ylim(0, 1)  # コサイン類似度は0から1の範囲
plt.grid(axis='y')

plt.tight_layout()  # レイアウトを調整
# plt.show()
# グラフをファイルに保存 (PNG形式)
plt.savefig("/home/s2410121/proj_LA/measure_similarities/llama3/llama3_barplot.png")
plt.close()

""" layerごとの絶対差の変遷のplot """

# 各モデルの平均絶対差を計算
mean_diffs = {
    'en_ja': [values['mean_absolute_difference'] for values in weight_changes_llama['en_ja'].values()],
    'en_ger': [values['mean_absolute_difference'] for values in weight_changes_llama['en_ger'].values()],
    'en_ita': [values['mean_absolute_difference'] for values in weight_changes_llama['en_ita'].values()],
    'en_ko': [values['mean_absolute_difference'] for values in weight_changes_llama['en_ko'].values()],
}

# 各リストの最小の長さを取得
min_length = min(len(mean_diffs[model]) for model in mean_diffs)

# 各モデルのリストを最小の長さに揃える
for model in mean_diffs:
    mean_diffs[model] = mean_diffs[model][:min_length]

# DataFrameに変換
mean_diffs_df = pd.DataFrame(mean_diffs)

# 可視化 (折れ線グラフ)
plt.figure(figsize=(12, 6))
for model in mean_diffs_df.columns:
    plt.plot(mean_diffs_df.index, mean_diffs_df[model], marker='o', label=model)

plt.xlabel('Layer Index')
plt.ylabel('Mean Absolute Difference')
plt.title('Mean Absolute Differences Across Models')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.show()
# グラフをファイルに保存 (PNG形式)
plt.savefig("/home/s2410121/proj_LA/measure_similarities/llama3/llama3_lineplot.png")
plt.close()
