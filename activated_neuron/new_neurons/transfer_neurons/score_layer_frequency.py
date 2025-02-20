import matplotlib.pyplot as plt
from collections import Counter

from funcs import unfreeze_pickle

def plot_layer_frequency(neuron_list):
    # layer_idx の出現回数をカウント
    layer_counts = Counter(layer_idx for layer_idx, _ in neuron_list)

    # データをソート（X軸を昇順に）
    sorted_layers = sorted(layer_counts.keys())
    frequencies = [layer_counts[layer] for layer in sorted_layers]

    # バープロットを作成
    plt.figure(figsize=(15, 15))
    plt.bar(sorted_layers, frequencies, color="blue", alpha=0.7)

    plt.xlabel("Layer Index", fontsize=30)
    plt.ylabel("Frequency", fontsize=30)
    plt.title("Neuron Frequency per Layer", fontsize=45)

    plt.xticks(sorted_layers, fontsize=25)
    plt.yticks(fontsize=25)

    plt.savefig(
        f'activated_neuron/new_neurons/images/transfers/distribution/{model}/layer_freq/{score_type}_n{n}',
        bbox_inches='tight',
        )

# params
model = 'llama3'# original llama
# model = 'llama' # <- llama learned L2.
# model = 'mistral'
langs = ['ja', 'nl', 'ko', 'it']
langs = ['ja']
score_types = ['cos_sim', 'L2_dis']
# score_types = ['cos_sim']
is_last_token_only = True
n = 1000
for L2 in langs:
    for score_type in score_types:
        # final scores.
        save_path_sorted_neurons = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/{score_type}/{L2}.pkl'
        sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)[:n]

        plot_layer_frequency(sorted_neurons)