import pandas as pd

from gpt2_visualize_blimp_eval_funcs import *

""" test """
# # CSVファイルをDataFrameとして読み込み
# file_path = "/home/s2410121/proj_LA/gpt2-small_blimp/blimp_evaluation_results_test.csv"
# # CSVファイルを読み込む
# # データの読み込み
# data = pd.read_csv(file_path)

# acc_comparison(data)
# multiple_models_acc_comparison(data)
# models_above_base_model(data)

""" gpt2 """
# file_path_gpt2_ALL = "/home/s2410121/proj_LA/blimp_evaluation_results_complete2_gpt2_all.csv"
# data_gpt2_ALL = pd.read_csv(file_path_gpt2_ALL)
# acc_comparison(data_gpt2_ALL, "gpt2", "gpt2_ALL")
# multiple_models_acc_comparison(data_gpt2_ALL, "gpt2", "gpt2_ALL")
# print(data_gpt2_ALL)
# data_gpt2_ALL['overall'] =

""" en_japanese """
# file_path_gpt2_en_ja = "/home/s2410121/proj_LA/gpt2-small_blimp/csv_files_final/blimp_evaluation_results_complete2_gpt2_all_final_en_ja.csv"
# data_gpt2_en_ja = pd.read_csv(file_path_gpt2_en_ja)
# acc_comparison(data_gpt2_en_ja, "gpt2", "gpt2_en_ja")
# multiple_models_acc_comparison(data_gpt2_en_ja, "gpt2", "gpt2_en_ja")
""" en_dutch """
# file_path_gpt2_en_du = "/home/s2410121/proj_LA/gpt2-small_blimp/csv_files/blimp_gpt2_en_du.csv"
# data_gpt2_en_du = pd.read_csv(file_path_gpt2_en_du)
# acc_comparison(data_gpt2_en_du, "gpt2", "gpt2_en_du")
# multiple_models_acc_comparison(data_gpt2_en_du, "gpt2", "gpt2_en_du")

""" modify_ja """
# file_path_gpt2_mo_ja = "/home/s2410121/proj_LA/gpt2-small_blimp/csv_files_final/blimp_evaluation_results_complete2_gpt2_all_final.csv"
# data_gpt2_mo_ja = pd.read_csv(file_path_gpt2_mo_ja)
# acc_comparison(data_gpt2_mo_ja, "gpt2", "gpt2_mo_ja")
# multiple_models_acc_comparison(data_gpt2_mo_ja, "gpt2", "gpt2_mo_ja")
# print(data_gpt2_ALL)

""" llama3 """
file_path_llama3_ALL = "/home/s2410121/proj_LA/gpt2-small_blimp/csv_files_final/blimp_evaluation_results_complete2_llama3_all_final.csv"
data_gpt2_llama3_ALL = pd.read_csv(file_path_llama3_ALL)
acc_comparison(data_gpt2_llama3_ALL, "llama3", "llama3_ALL")
multiple_models_acc_comparison(data_gpt2_llama3_ALL, "llama3", "llama3_ALL")
"""en_japanese"""
file_path_llama3_en_ja = "/home/s2410121/proj_LA/gpt2-small_blimp/csv_files_final/blimp_evaluation_results_complete2_llama3_all_fina_en_ja.csv"
data_llama3_en_ja = pd.read_csv(file_path_llama3_en_ja)
acc_comparison(data_llama3_en_ja, "llama3", "llama3_en_ja")
multiple_models_acc_comparison(data_llama3_en_ja, "llama3", "llama3_en_ja")
models_above_base_model(data_llama3_en_ja, "llama3", "llama3_en_ja")
