from measure_similarities import *

import numpy as np
import matplotlib.pyplot as plt

import os

# 保存先ディレクトリを設定
output_dir = './weight_change_plots/'
os.makedirs(output_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成

""" 可視化 """
# 全ての言語ペアに対してループ
for pair, weight_changes_pair in weight_changes_llama.items():

    # データの準備
    layers = list(weight_changes_pair.keys())  # レイヤー名
    cos_sim_values = [weight_changes_pair[layer]['cosine_similarity'] for layer in layers]  # コサイン類似度
    mean_abs_diff_values = [weight_changes_pair[layer]['mean_absolute_difference'] for layer in layers]  # 絶対差の平均

    # X軸用のインデックス
    x = np.arange(len(layers))

    # グラフのサイズ
    plt.figure(figsize=(12, 6))

    # コサイン類似度をバーグラフで表示
    plt.bar(x - 0.2, cos_sim_values, width=0.4, label='Cosine Similarity', color='b')

    # 絶対差の平均をバーグラフで表示
    plt.bar(x + 0.2, mean_abs_diff_values, width=0.4, label='Mean Absolute Difference', color='g')

    # レイヤー名をX軸に設定
    plt.xticks(x, layers, rotation=90, fontsize=8)

    # グラフのタイトル、ラベルを設定
    plt.title(f'Weight Changes Across Layers ({pair})')
    plt.xlabel('Layers')
    plt.ylabel('Values')

    # 凡例を追加
    plt.legend()

    # グラフを表示 (省略可能)
    # plt.show()

    # グラフをファイルに保存
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'weight_changes_{pair}.png'))  # ファイル名に言語ペアを含む
    plt.close()  # メモリ節約のためにグラフを閉じる
