""" overallをみる """

import pandas as pd

# CSVファイルを読み込んでデータフレームを作成
path = "/home/s2410121/proj_LA/activated_neuron/new_neurons/intervention/csv/llama3/act_sum/blimp/0.01_th/shared_same_semantics/all/en_ja.csv"
df = pd.read_csv(path)

# 各モデルごとに正答率の平均を計算します
overall_accuracy = df.groupby('Model')['Accuracy'].mean().reset_index()

# 列名を変更してOVERALLにします
overall_accuracy.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)

# 結果を表示
print(overall_accuracy)
