import numpy as np
from scipy.stats import f
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
from funcs import unfreeze_pickle, unfreeze_np_arrays, save_as_pickle

langs = ['ja', 'nl', 'ko', 'it']
model_types = ['llama3', 'mistral', 'aya']
score_types = ['cos_sim', 'L2_dis']
score_types = ['cos_sim']
is_reverses = [False, True]
results = defaultdict(list)

def welch_t_test(labels, values):
    group0 = values[labels == 0]
    group1 = values[labels == 1]

    # Welch t test（equal_var=False -> 等分散を仮定しない）.
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

def mann_whitney_u_test(labels, values):
    group0 = values[labels == 0]
    group1 = values[labels == 1]

    stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')

    return stat, p_value

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
# lang_family.
labels_dict = {
    'ja': l1 + l2 + l1 + l2 + l2,
    'nl': l2 + l1 + l2 + l1 + l1,
    'ko': l1 + l2 + l1 + l2 + l2,
    'it': l2 + l1 + l2 + l1 + l1,
    'en': l2 + l1 + l2 + l1 + l1,
}

top_n = 1000
significance_level = 0.05

for model_type in model_types:
    for is_reverse in is_reverses:
        for score_type in score_types:
            key_prefix = f"{model_type}_{'type-2' if is_reverse else 'type-1'}"
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
                    # eta, F_val, p_val = compute_eta_squared_and_f(labels_list, vals) # ANOVA
                    # eta, F_val, p_val = welch_t_test(labels_list, vals) # t test(welch)
                    stat, p_val = mann_whitney_u_test(labels_list, vals) # Mann-Whitney U test.
                    # eta_list.append(eta)
                    p_list.append(p_val)

                # eta_arr = np.array(eta_list)
                p_arr = np.array(p_list)

                # mean_eta = np.mean(eta_arr)
                significant_count = np.sum(p_arr < significance_level)
                # print(f"mean η² = {mean_eta:.4f}")
                print(f"{significant_count}/{top_n} neurons are SIGNIFICANT at α={significance_level}\n")
                key = f"{key_prefix}_{L2}"
                proportion_significant = significant_count / top_n
                results[key] = [proportion_significant]

# save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/corr_ratio/h_test/all.pkl' # welch t test.
# save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/corr_ratio/h_test/all_anova.pkl' # anova.
# save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/corr_ratio/h_test/all_man_whitney.pkl' # mann-whitney u test.

# lang family.
# save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/corr_ratio/h_test/all_lang_family.pkl' # welch t test.
# save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/corr_ratio/h_test/all_anova_lang_family.pkl' # anova.
save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/corr_ratio/h_test/all_man_whitney_lang_family.pkl'
save_as_pickle(save_path, results)

results = unfreeze_pickle(save_path)


plt.rcParams["font.family"] = "DejaVu Serif"

fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor('#f7f7f7')

bar_width = 0.12
x = np.arange(len(langs))

model_types = ['llama3', 'mistral', 'aya']
is_reverses = [False, True]

n_groups = len(model_types) * len(is_reverses)
offsets = np.linspace(-bar_width * n_groups / 2, bar_width * n_groups / 2, n_groups)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

idx = 0
for model_type in model_types:
    model_title = 'LLaMA3-8B' if model_type == 'llama3' else 'Mistral-7B' if model_type == 'mistral' else 'Aya expanse-8B'

    for is_reverse in is_reverses:
        key_prefix = f"{model_type}_{'type-2' if is_reverse else 'type-1'}"
        y = [results[f"{key_prefix}_{L2}"][0] for L2 in langs]
        label_name = f"{model_title} {'Type2' if is_reverse else 'Type1'}"

        ax.bar(
            x + offsets[idx], y, width=bar_width, label=label_name,
            color=colors[idx % len(colors)],
            alpha=0.85,
            edgecolor='black',
            linewidth=0.7,
            capstyle='round'
        )
        idx += 1

ax.set_xticks(x)
ax.set_xticklabels(langs, fontsize=40)
ax.tick_params(axis='y', labelsize=25)
ax.set_ylabel('Proportion of Significant Neurons', fontsize=30)
ax.set_ylim(0, 1)
ax.set_title('Proportion of Significant Neurons', fontsize=40)
ax.grid(axis='y', linestyle='--', alpha=0.5)

legend = ax.legend(frameon=True, shadow=True, fontsize=30)
legend.get_frame().set_edgecolor('gray')
legend.get_frame().set_linewidth(1)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color('gray')

# path = '/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/corr_ratio/h_test/all_models'
# path = '/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/corr_ratio/h_test/all_models_anova'
# path = '/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/corr_ratio/h_test/all_models_mann_whitney'
# lang family
# path = '/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/corr_ratio/h_test/all_models_lang_family'
# path = '/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/corr_ratio/h_test/all_models_anova_lang_family'
path = '/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/corr_ratio/h_test/all_models_ang_family_mann_whitney'
with PdfPages(path + '.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0.01)