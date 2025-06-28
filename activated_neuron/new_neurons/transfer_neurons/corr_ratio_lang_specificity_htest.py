import numpy as np
from scipy.stats import f
from scipy.stats import ttest_ind
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

from funcs import unfreeze_pickle, unfreeze_np_arrays

langs = ['ja', 'nl', 'ko', 'it']
model_types = ['llama3', 'mistral', 'aya']
score_types = ['cos_sim', 'L2_dis']
score_types = ['cos_sim']
is_reverses = [False, True]

def welch_t_test(labels, values):
    group0 = values[labels == 0]
    group1 = values[labels == 1]

    # Welch の t検定（等分散を仮定しない）
    t_stat, p_value = ttest_ind(group0, group1, equal_var=False)
    
    # calc correlation ratio.
    eta_squared = correlationRatio(labels, values)
    
    return eta_squared, t_stat, p_value

def correlationRatio(categories, values):
    interclass_variation  = sum([
        (len(values[categories == i]) * ((values[categories == i].mean() - values.mean()) ** 2)).sum() for i in np.unique(categories)
    ]) 
    total_variation = sum((values - values.mean()) ** 2)
    return interclass_variation / total_variation

def compute_eta_squared_and_f(categories, values):
    cats = np.unique(categories)
    n_total = len(values)
    n_groups = len(cats)

    group_means = np.array([values[categories == c].mean() for c in cats])
    group_counts = np.array([np.sum(categories == c) for c in cats])
    overall_mean = values.mean()

    ss_between = np.sum(group_counts * (group_means - overall_mean)**2)
    ss_within = 0
    for c in cats:
        group_vals = values[categories == c]
        ss_within += np.sum((group_vals - group_vals.mean())**2)

    df_between = n_groups - 1
    df_within = n_total - n_groups

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    F = ms_between / ms_within if ms_within != 0 else 0
    eta_squared = ss_between / (ss_between + ss_within) if (ss_between + ss_within) != 0 else 0
    p_value = 1 - f.cdf(F, df_between, df_within) if F != 0 else 1.0

    return eta_squared, F, p_value

# ラベル設定
l1 = [1] * 1000
l2 = [0] * 1000

labels_dict = {
    'ja': l1 + l2 + l2 + l2 + l2,
    'nl': l2 + l1 + l2 + l2 + l2,
    'ko': l2 + l2 + l1 + l2 + l2,
    'it': l2 + l2 + l2 + l1 + l2,
    'en': l2 + l2 + l2 + l2 + l1,
}

top_n = 1000
significance_level = 0.05

for model_type in model_types:
    for is_reverse in is_reverses:
        for score_type in score_types:
            for L2 in langs:
                save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/activations/{L2}_last_token.npz"
                activations_arr = unfreeze_np_arrays(save_path_activations)

                if is_reverse:
                    save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
                    sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
                    sorted_neurons = [n for n in sorted_neurons if n[0] in (range(20, 32) if model_type != 'bloom' else range(20, 30))]
                else:
                    save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
                    sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
                    sorted_neurons = [n for n in sorted_neurons if n[0] in range(20)]

                labels_list = np.array(labels_dict[L2])

                print(f"--- {model_type} {'type-2' if is_reverse else 'type-1'}, lang={L2}, score={score_type} ---")

                eta_list = []
                p_list = []
                for (layer_i, neuron_i) in sorted_neurons[:top_n]:
                    vals = activations_arr[layer_i, neuron_i, :]
                    # eta, F_val, p_val = compute_eta_squared_and_f(labels_list, vals)
                    eta, F_val, p_val = welch_t_test(labels_list, vals)
                    eta_list.append(eta)
                    p_list.append(p_val)

                eta_arr = np.array(eta_list)
                p_arr = np.array(p_list)

                mean_eta = np.mean(eta_arr)
                significant_count = np.sum(p_arr < significance_level)
                print(f"mean η² = {mean_eta:.4f}")
                print(f"{significant_count}/{top_n} neurons are SIGNIFICANT at α={significance_level}\n")