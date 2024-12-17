import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" check OVERALL """
df_shared = pd.read_csv('/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/shared/n_2000/llama3_en_it.csv')
df_normal_COMP = pd.read_csv("/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/normal_COMP/n_2000/blimp_eval_llama3_en_it_COMP.csv")
df_L1_or_L2 = pd.read_csv("/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/L1_or_L2/n_2000/blimp_eval_llama3_en_it_L1_or_L2.csv")
df_L1_specific = pd.read_csv("/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/L1_specific/n_2000/llama3_en_it_L1_specific.csv")

acc_overall_shared = df_shared.groupby('Model')['Accuracy'].mean().reset_index()
acc_overall_shared.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
print(acc_overall_shared)
acc_overall_normal_COMP = df_normal_COMP.groupby('Model')['Accuracy'].mean().reset_index()
acc_overall_normal_COMP.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
print(acc_overall_normal_COMP)
acc_overall_L1_or_L2 = df_L1_or_L2.groupby('Model')['Accuracy'].mean().reset_index()
acc_overall_L1_or_L2.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
print(acc_overall_L1_or_L2)
acc_overall_L1_specific = df_L1_specific.groupby('Model')['Accuracy'].mean().reset_index()
acc_overall_L1_specific.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)
print(acc_overall_L1_specific)

