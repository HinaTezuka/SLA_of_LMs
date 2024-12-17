""" overallをみる """

import pandas as pd

# CSVファイルを読み込んでデータフレームを作成
# df = pd.read_csv('/home/s2410121/proj_LA/gpt2-small_blimp/csv_files/blimp_evaluation_results_complete_en_ger_ita_fre_ko_spa.csv')
# df = pd.read_csv("/home/s2410121/proj_LA/gpt2-small_blimp/csv_files/blimp_llama3_en_ja.csv")
# df = pd.read_csv("/home/s2410121/proj_LA/gpt2-small_blimp/csv_files/blimp_llama3_en_ja.csv")
# df = pd.read_csv("/home/s2410121/proj_LA/blimp_evaluation_results_complete2_llama3_all_final.csv")
df = pd.read_csv("/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/jblimp/shared/all/llama3_en_ja_shared_non_translation.csv")

# 各モデルごとに正答率の平均を計算します
overall_accuracy = df.groupby('Model')['Accuracy'].mean().reset_index()

# 列名を変更してOVERALLにします
overall_accuracy.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)

# 結果を表示
print(overall_accuracy)


