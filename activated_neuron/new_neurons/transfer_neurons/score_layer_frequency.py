import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter

from funcs import unfreeze_pickle

def plot_layer_frequency(neuron_list: list, is_reverse: bool):
    # layer_idx の出現回数をカウント
    layer_counts = Counter(layer_idx for layer_idx, _ in neuron_list)

    # データをソート（X軸を昇順に）
    sorted_layers = sorted(layer_counts.keys())
    frequencies = [layer_counts[layer] for layer in sorted_layers]
    display_layers = [layer + 1 for layer in sorted_layers]

    plt.figure(figsize=(15, 15))
    plt.bar(display_layers, frequencies, color="blue", alpha=0.7)

    plt.xlabel("Layer Index", fontsize=30)
    plt.ylabel("Frequency", fontsize=30)
    plt.title("Neuron Frequency per Layer", fontsize=45)

    plt.xticks(display_layers, fontsize=25)
    plt.yticks(fontsize=25)

    if not is_reverse:
        save_path = f'activated_neuron/new_neurons/images/transfers/distribution/{model}/layer_freq/{score_type}_{L2}_n{n}'
    else:
        save_path = f'activated_neuron/new_neurons/images/transfers/distribution/{model}/layer_freq/reverse/{score_type}_{L2}_n{n}'        
    # plt.savefig(
    #     save_path,
    #     bbox_inches='tight',
    #     )
    with PdfPages(save_path + '.pdf') as pdf:
        pdf.savefig(bbox_inches='tight', pad_inches=0.01)
        plt.close()

if __name__ == '__main__':
    models = ['llama3', 'mistral', 'aya']
    models = ['mistral']
    langs = ['ja', 'nl', 'ko', 'it']
    score_types = ['cos_sim', 'L2_dis']
    is_last_token_only = True
    nums = [100, 1000, 3000, 5000, 10000]
    is_reverses = [True, False]
    for model in models:
        for is_reverse in is_reverses:
            for L2 in langs:
                for score_type in score_types:
                    for n in nums:
                        if is_reverse:
                            save_path_sorted_neurons = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl'
                            sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
                            sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
                        elif not is_reverse:
                            save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/{score_type}/{L2}_mono_train.pkl"
                            sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
                            sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20)]]
                        sorted_neurons = sorted_neurons[:n]

                        plot_layer_frequency(sorted_neurons, is_reverse)