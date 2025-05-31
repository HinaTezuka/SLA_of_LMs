import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter

from funcs import unfreeze_pickle

def plot_layer_frequency(neuron_list: list, is_reverse: bool):
    layer_counts = Counter(layer_idx for layer_idx, _ in neuron_list)

    sorted_layers = sorted(layer_counts.keys())
    frequencies = [layer_counts[layer] for layer in sorted_layers]
    display_layers = [layer + 1 for layer in sorted_layers]

    plt.figure(figsize=(15, 15))
    plt.bar(display_layers, frequencies, color="blue", alpha=0.7)

    plt.xlabel("Layer Index", fontsize=45)
    plt.ylabel("Frequency", fontsize=45)
    plt.title("Neuron Frequency per Layer", fontsize=55)

    xtick_min = min(display_layers)
    xtick_max = max(display_layers)
    xtick_step = 5
    xticks = list(range(xtick_min, xtick_max + 1, xtick_step))
    plt.xticks(xticks, fontsize=35)
    plt.xticks(display_layers, fontsize=35)
    plt.yticks(fontsize=35)

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


def plot_layer_distribution_all_langs(all_neurons_by_lang: dict, is_reverse: bool, model: str, score_type: str, L2s: list, n: int):
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.figure(figsize=(15, 15))
    palette = sns.color_palette('Set2', len(all_neurons_by_lang))
    for (lang, neuron_list), color in zip(all_neurons_by_lang.items(), palette):
        layer_indices = [layer for layer, _ in neuron_list]
        if layer_indices:
            sns.kdeplot(
                layer_indices,
                bw_adjust=0.5,
                label=lang,
                linewidth=3,
                fill=True,
                alpha=0.1,
                color=color,
            )

    plt.xlabel("Layer Index", fontsize=45)
    plt.ylabel("Density", fontsize=45)
    title = 'Type-1 Neurons' if not is_reverse else 'Type-2 Neurons'
    plt.title(f"{title} (n={n})", fontsize=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(title="Language", fontsize=50, title_fontsize=45)

    if not is_reverse:
        save_path = f'activated_neuron/new_neurons/images/transfers/distribution/{model}/layer_freq/{score_type}_allLangs_n{n}'
    else:
        save_path = f'activated_neuron/new_neurons/images/transfers/distribution/{model}/layer_freq/reverse/{score_type}_allLangs_n{n}'

    with PdfPages(save_path + '.pdf') as pdf:
        pdf.savefig(bbox_inches='tight', pad_inches=0.01)
        plt.close()

if __name__ == '__main__':
    models = ['llama3', 'mistral', 'aya']
    langs = ['ja', 'nl', 'ko', 'it']
    score_types = ['cos_sim', 'L2_dis']
    nums = [100, 1000, 3000, 5000, 10000]
    is_reverses = [True, False]

    for model in models:
        for is_reverse in is_reverses:
            for score_type in score_types:
                for n in nums:
                    all_neurons_by_lang = {}
                    for L2 in langs:
                        if is_reverse:
                            path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl'
                            sorted_neurons = unfreeze_pickle(path)
                            sorted_neurons = [neuron for neuron in sorted_neurons if 20 <= neuron[0] < 32]
                        else:
                            path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/{score_type}/{L2}_mono_train.pkl"
                            sorted_neurons = unfreeze_pickle(path)
                            sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] < 20]

                        sorted_neurons = sorted_neurons[:n]
                        all_neurons_by_lang[L2] = sorted_neurons

                    plot_layer_distribution_all_langs(all_neurons_by_lang, is_reverse, model, score_type, langs, n)

# if __name__ == '__main__':
#     models = ['llama3', 'mistral', 'aya']
#     langs = ['ja', 'nl', 'ko', 'it']
#     score_types = ['cos_sim', 'L2_dis']
#     is_last_token_only = True
#     nums = [100, 1000, 3000, 5000, 10000]
#     is_reverses = [True, False]
#     for model in models:
#         for is_reverse in is_reverses:
#             for L2 in langs:
#                 for score_type in score_types:
#                     for n in nums:
#                         if is_reverse:
#                             save_path_sorted_neurons = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl'
#                             sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
#                             sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
#                         elif not is_reverse:
#                             save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model}/final_scores/{score_type}/{L2}_mono_train.pkl"
#                             sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
#                             sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20)]]
#                         sorted_neurons = sorted_neurons[:n]

#                         plot_layer_distribution_all_langs(sorted_neurons, is_reverse)