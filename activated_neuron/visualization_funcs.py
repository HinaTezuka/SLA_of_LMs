import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/detect_act_neurons.py")
sys.path.append("/home/s2410121/proj_LA/activated_neuron/shared_neurons.py")

import numpy as np
import matplotlib.pyplot as plt

nums_of_neurons_llama3 = 14336 # nums of all neurons (LLaMA-3-8B: MLP)

def visualize_neurons_with_line_plot_mean(
                                    L1,
                                    L2,
                                    # main
                                    activated_neurons_L1,
                                    activated_neurons_L2,
                                    shared_neurons,
                                    specific_neurons_L1,
                                    specific_neurons_L2,
                                    folder: str,
                                    # base
                                    shared_neurons_base,
                                    ):
    # nums of all layers(LLaMA-3-8B)
    num_layers = 32

    """ main """
    L2_counts = [0] * num_layers
    L1_counts = [0] * num_layers
    shared_counts = [0] * num_layers
    specific_L2_counts = [0] * num_layers
    specific_L1_counts = [0] * num_layers
    """ base line """
    shared_counts_base = [0] * num_layers

    shared_counts_std = [0] * num_layers  # Standard deviation for shared neurons

    # counting activate/non-activate counts(culc mean)
    for layer_idx in range(num_layers):
        L1_counts[layer_idx] = np.array(activated_neurons_L1[layer_idx]).mean()  # mean nums for L1(en) neurons
        L2_counts[layer_idx] = np.array(activated_neurons_L2[layer_idx]).mean()  # mean nums for L2 neurons
        shared_counts[layer_idx] = np.array(shared_neurons[layer_idx]).mean()  # shared_neurons
        specific_L1_counts[layer_idx] = np.array(specific_neurons_L1[layer_idx]).mean()  # specific neurons for L1(en)
        specific_L2_counts[layer_idx] = np.array(specific_neurons_L2[layer_idx]).mean()  # specific neurons for L2

        """ base line """
        shared_counts_base[layer_idx] = np.array(shared_neurons_base[layer_idx]).mean()  # shared_neurons

        """ Calculate the standard deviation for shared neurons (used for error bars) """
        shared_counts_std[layer_idx] = np.array(shared_neurons[layer_idx]).std()

    # plot
    plt.figure(figsize=(15, 10))

    # L2 and L1 activated neurons with original colors (transparent)
    plt.plot(range(num_layers), L2_counts, label=f'{L2} Activated Neurons', marker='o', alpha=0.5)
    plt.plot(range(num_layers), L1_counts, label=f'{L1} Activated Neurons', marker='o', alpha=0.5)

    # Add error bars for shared counts (dashed line style for the error bars)
    plt.errorbar(range(num_layers), shared_counts, yerr=shared_counts_std, label='Shared Neurons(sentence pair of same meanings)',
                fmt='o', markersize=6, linestyle='-', linewidth=4, capsize=5)

    # Specific neurons with normal line style
    plt.plot(range(num_layers), specific_L2_counts, label=f'Specific to {L2}', marker='o', alpha=0.5)
    plt.plot(range(num_layers), specific_L1_counts, label=f'Specific to {L1}', marker='o', alpha=0.5)

    # Base line with a distinct style
    plt.plot(range(num_layers), shared_counts_base, label='Shared Neurons(base: sentence pair of different meanings)', marker='x', linewidth=2)

    plt.title(f'Neuron Activation Counts per Layer ({L1} and {L2})')
    plt.xlabel('Layer Index')
    plt.ylabel('Number of Neurons')
    plt.xticks(range(num_layers))
    plt.legend()
    plt.grid()

    # グラフの保存
    plt.savefig(f'/home/s2410121/proj_LA/activated_neuron/images/means/{folder}/activated_neuron_{L1}_{L2}.png')
    plt.close()

# def visualize_neurons_with_line_plot_mean(
#                                     L1,
#                                     L2,
#                                     # main
#                                     activated_neurons_L1,
#                                     activated_neurons_L2,
#                                     shared_neurons,
#                                     specific_neurons_L1,
#                                     specific_neurons_L2,
#                                     folder: str,
#                                     # base
#                                     shared_neurons_base,
#                                     ):
#     # nums of all layers(LLaMA-3-8B)
#     num_layers = 32

#     """ main """
#     L2_counts = [0] * num_layers
#     L1_counts = [0] * num_layers
#     shared_counts = [0] * num_layers
#     specific_L2_counts = [0] * num_layers
#     specific_L1_counts = [0] * num_layers
#     """ base line """
#     shared_counts_base = [0] * num_layers

#     # counting activate/non-activate counts(culc mean)
#     for layer_idx in range(num_layers):
#         L1_counts[layer_idx] = np.array(activated_neurons_L1[layer_idx]).mean() # mean nums for L1(en) neurons
#         L2_counts[layer_idx] = np.array(activated_neurons_L2[layer_idx]).mean()  # mean nums for L2 neurons
#         shared_counts[layer_idx] = np.array(shared_neurons[layer_idx]).mean()  # shared_neurons
#         specific_L1_counts[layer_idx] = np.array(specific_neurons_L1[layer_idx]).mean()  # specific neurons for L1(en)
#         specific_L2_counts[layer_idx] = np.array(specific_neurons_L2[layer_idx]).mean()  # specific neurons for L2

#         """ base line """
#         shared_counts_base[layer_idx] = np.array(shared_neurons_base[layer_idx]).mean()  # shared_neurons

#     # plot
#     plt.figure(figsize=(15, 10))
#     plt.plot(range(num_layers), L2_counts, label=f'{L2} Activated Neurons', marker='o')
#     plt.plot(range(num_layers), L1_counts, label=f'{L1} Activated Neurons', marker='o')
#     plt.plot(range(num_layers), shared_counts, label='Shared Neurons(sentence pair of same meanings)', marker='o', linewidth=6)
#     plt.plot(range(num_layers), specific_L2_counts, label=f'Specific to {L2}', marker='o')
#     plt.plot(range(num_layers), specific_L1_counts, label=f'Specific to {L1}', marker='o')

#     """ base line """
#     plt.plot(range(num_layers), shared_counts_base, label='Shared Neurons(base)', marker='x', linewidth=2)

#     plt.title(f'Neuron Activation Counts per Layer ({L1} and {L2})')
#     plt.xlabel('Layer Index')
#     plt.ylabel('Number of Neurons')
#     plt.xticks(range(num_layers))
#     plt.legend()
#     plt.grid()

#     # グラフの保存
#     plt.savefig(f'/home/s2410121/proj_LA/activated_neuron/images/means/{folder}/activated_neuron_{L1}_{L2}.png')
#     plt.close()

def visualize_neurons_with_line_plot_simple(
                                    L1,
                                    L2,
                                    # main
                                    activated_neurons_L1,
                                    activated_neurons_L2,
                                    non_activated_neurons_L1,
                                    non_activated_neurons_L2,
                                    shared_neurons,
                                    specific_neurons_L1,
                                    specific_neurons_L2,
                                    non_activated_neurons_all,
                                    folder: str,
                                    # base
                                    shared_neurons_base,
                                    ):
    # nums of all layers(LLaMA-3-8B)
    num_layers = 32

    """ main """
    L2_counts = [0] * num_layers
    L1_counts = [0] * num_layers
    shared_counts = [0] * num_layers
    specific_L2_counts = [0] * num_layers
    specific_L1_counts = [0] * num_layers
    non_activated_L2_counts = [0] * num_layers  # 日本語の非活性化ニューロン
    non_activated_L1_counts = [0] * num_layers  # 英語の非活性化ニューロン
    non_activated_all_counts = [0] * num_layers  # 両言語の非活性化共通ニューロン
    """ base line """
    shared_counts_base = [0] * num_layers

    # counting activate/non-activate counts
    for layer_idx in range(num_layers):

        L1_counts[layer_idx] = len(activated_neurons_L1[layer_idx])  # nums for ja neurons
        L2_counts[layer_idx] = len(activated_neurons_L2[layer_idx])  # nums for en neurons
        shared_counts[layer_idx] = len(shared_neurons[layer_idx])  # shared_neurons
        specific_L2_counts[layer_idx] = len(specific_neurons_L2[layer_idx])  # specific neurons for ja
        specific_L1_counts[layer_idx] = len(specific_neurons_L1[layer_idx])  # specific neurons for en
        # non_activated_L2_counts[layer_idx] = len(non_activated_neurons_L2[layer_idx])  # non-activate ja
        # non_activated_L1_counts[layer_idx] = len(non_activated_neurons_L1[layer_idx])  # non-activate en
        # non_activated_all_counts[layer_idx] = len(non_activated_neurons_all[layer_idx])  # non-activate for both
        """ base line """
        shared_counts_base[layer_idx] = len(shared_neurons_base[layer_idx])  # shared_neurons
    # plot
    plt.figure(figsize=(15, 10))
    plt.plot(range(num_layers), L2_counts, label=f'{L2} Activated Neurons', marker='o')
    plt.plot(range(num_layers), L1_counts, label=f'{L1} Activated Neurons', marker='o')
    plt.plot(range(num_layers), shared_counts, label='Shared Neurons(sentence pair of same meanings)', marker='o', linewidth=6)
    plt.plot(range(num_layers), specific_L2_counts, label=f'Specific to {L2}', marker='o')
    plt.plot(range(num_layers), specific_L1_counts, label=f'Specific to {L1}', marker='o')
    # plt.plot(range(num_layers), non_activated_L2_counts, label=f'Non-Activated {L2} Neurons', marker='x', linestyle='--')
    # plt.plot(range(num_layers), non_activated_L1_counts, label=f'Non-Activated {L1} Neurons', marker='x', linestyle='--')
    # plt.plot(range(num_layers), non_activated_all_counts, label='Non-Activated Neurons (Both)', marker='s', linestyle='-.')
    """ base line """
    plt.plot(range(num_layers), shared_counts_base, label='Shared Neurons(base)', marker='x', linewidth=2)

    plt.title(f'Neuron Activation Counts per Layer ({L1} and {L2})')
    plt.xlabel('Layer Index')
    plt.ylabel('Number of Neurons')
    plt.xticks(range(num_layers))
    plt.legend()
    plt.grid()

    # グラフの保存
    plt.savefig(f'/home/s2410121/proj_LA/activated_neuron/images/{folder}/activated_neuron_{L1}_{L2}.png')
    plt.close()

def visualize_neurons_with_line_plot(
                                    L1,
                                    L2,
                                    # main
                                    activated_neurons_L1,
                                    activated_neurons_L2,
                                    non_activated_neurons_L1,
                                    non_activated_neurons_L2,
                                    shared_neurons,
                                    specific_neurons_L1,
                                    specific_neurons_L2,
                                    non_activated_neurons_all,
                                    folder: str,
                                    # base
                                    shared_neurons_base,
                                    ):
    # nums of all layers(LLaMA-3-8B)
    num_layers = 32

    # list for aggregating all types of neurons
    """ main """
    L2_counts = [0] * num_layers
    L1_counts = [0] * num_layers
    shared_counts = [0] * num_layers
    specific_L2_counts = [0] * num_layers
    specific_L1_counts = [0] * num_layers
    non_activated_L2_counts = [0] * num_layers  # 日本語の非活性化ニューロン
    non_activated_L1_counts = [0] * num_layers  # 英語の非活性化ニューロン
    non_activated_all_counts = [0] * num_layers  # 両言語の非活性化共通ニューロン
    """ base line """
    shared_counts_base = [0] * num_layers

    # counting activate/non-activate counts
    for layer_idx in range(num_layers):
        # activated_neurons_L1 と activated_neurons_L2 から、各層のユニークなニューロン数を取得
        L1_neurons_set = set(activated_neurons_L1[layer_idx][1][:, 2])  # 第3列がニューロンインデックス
        L2_neurons_set = set(activated_neurons_L2[layer_idx][1][:, 2])

        L2_counts[layer_idx] = len(L2_neurons_set)  # nums for ja neurons
        L1_counts[layer_idx] = len(L1_neurons_set)  # nums for en neurons
        shared_counts[layer_idx] = len(set(shared_neurons[layer_idx][1].flatten()))  # shared_neurons
        specific_L2_counts[layer_idx] = len(set(specific_neurons_L2[layer_idx][1].flatten()))  # specific neurons for ja
        specific_L1_counts[layer_idx] = len(set(specific_neurons_L1[layer_idx][1].flatten()))  # specific neurons for en
        # non_activated_L2_counts[layer_idx] = len(set(non_activated_neurons_L2[layer_idx][1].flatten()))  # non-activate ja
        # non_activated_L1_counts[layer_idx] = len(set(non_activated_neurons_L1[layer_idx][1].flatten()))  # non-activate en
        # non_activated_all_counts[layer_idx] = len(set(non_activated_neurons_all[layer_idx][1].flatten()))  # non-activate for both
        """ base line """
        shared_counts_base[layer_idx] = len(set(shared_neurons_base[layer_idx][1].flatten()))  # shared_neurons

    # plot
    plt.figure(figsize=(15, 10))
    plt.plot(range(num_layers), L2_counts, label=f'{L2} Activated Neurons', marker='o')
    plt.plot(range(num_layers), L1_counts, label=f'{L1} Activated Neurons', marker='o')
    plt.plot(range(num_layers), shared_counts, label='Shared Neurons', marker='o', linewidth=6)
    plt.plot(range(num_layers), specific_L2_counts, label=f'Specific to {L2}', marker='o')
    plt.plot(range(num_layers), specific_L1_counts, label=f'Specific to {L1}', marker='o')
    # plt.plot(range(num_layers), non_activated_L2_counts, label=f'Non-Activated {L2} Neurons', marker='x', linestyle='--')
    # plt.plot(range(num_layers), non_activated_L1_counts, label=f'Non-Activated {L1} Neurons', marker='x', linestyle='--')
    # plt.plot(range(num_layers), non_activated_all_counts, label='Non-Activated Neurons (Both)', marker='s', linestyle='-.')
    # base line
    plt.plot(range(num_layers), shared_counts_base, label='Shared Neurons(base)', marker='x', linestyle='-.')

    plt.title(f'Neuron Activation Counts per Layer ({L1} and {L2})')
    plt.xlabel('Layer Index')
    plt.ylabel('Number of Neurons')
    plt.xticks(range(num_layers))
    plt.legend()
    plt.grid()

    # グラフの保存
    plt.savefig(f'/home/s2410121/proj_LA/activated_neuron/images/{folder}/activated_neuron_{L1}_{L2}.png')
    plt.close()


    # list for aggregating all types of neurons
    """ main """
    L2_counts = [0] * num_layers
    L1_counts = [0] * num_layers
    shared_counts = [0] * num_layers
    specific_L2_counts = [0] * num_layers
    specific_L1_counts = [0] * num_layers
    non_activated_L2_counts = [0] * num_layers  # 日本語の非活性化ニューロン
    non_activated_L1_counts = [0] * num_layers  # 英語の非活性化ニューロン
    non_activated_all_counts = [0] * num_layers  # 両言語の非活性化共通ニューロン
    """ base line """
    shared_counts_base = [0] * num_layers

    # counting activate/non-activate counts
    for layer_idx in range(num_layers):
        # activated_neurons_L1 と activated_neurons_L2 から、各層のユニークなニューロン数を取得
        L1_neurons_set = set(activated_neurons_L1[layer_idx][1][:, 2].flatten())  # 第3列がニューロンインデックス
        L2_neurons_set = set(activated_neurons_L2[layer_idx][1][:, 2].flatten())

        L2_counts[layer_idx] = len(L2_neurons_set)  # nums for ja neurons
        L1_counts[layer_idx] = len(L1_neurons_set)  # nums for en neurons
        shared_counts[layer_idx] = len(set(shared_neurons[layer_idx][1].flatten()))  # shared_neurons
        specific_L2_counts[layer_idx] = len(set(specific_neurons_L2[layer_idx][1].flatten()))  # specific neurons for ja
        specific_L1_counts[layer_idx] = len(set(specific_neurons_L1[layer_idx][1].flatten()))  # specific neurons for en
        # non_activated_L2_counts[layer_idx] = len(set(non_activated_neurons_L2[layer_idx][1].flatten()))  # non-activate ja
        # non_activated_L1_counts[layer_idx] = len(set(non_activated_neurons_L1[layer_idx][1].flatten()))  # non-activate en
        # non_activated_all_counts[layer_idx] = len(set(non_activated_neurons_all[layer_idx][1].flatten()))  # non-activate for both
        """ base line """
        shared_counts_base[layer_idx] = len(set(shared_neurons_base[layer_idx][1].flatten()))  # shared_neurons
        # print(f'base:main {shared_counts_base[layer_idx], shared_counts[layer_idx]}')

    # plot
    plt.figure(figsize=(15, 10))
    plt.plot(range(num_layers), L2_counts, label=f'{L2} Activated Neurons', marker='o')
    plt.plot(range(num_layers), L1_counts, label=f'{L1} Activated Neurons', marker='o')
    plt.plot(range(num_layers), shared_counts, label='Shared Neurons(sentence pair of same meanings)', marker='o', linewidth=6)
    plt.plot(range(num_layers), specific_L2_counts, label=f'Specific to {L2}', marker='o')
    plt.plot(range(num_layers), specific_L1_counts, label=f'Specific to {L1}', marker='o')
    # plt.plot(range(num_layers), non_activated_L2_counts, label=f'Non-Activated {L2} Neurons', marker='x', linestyle='--')
    # plt.plot(range(num_layers), non_activated_L1_counts, label=f'Non-Activated {L1} Neurons', marker='x', linestyle='--')
    # plt.plot(range(num_layers), non_activated_all_counts, label='Non-Activated Neurons (Both)', marker='s', linestyle='-.')
    """ base line """
    plt.plot(range(num_layers), shared_counts_base, label='Shared Neurons(base)', marker='x', linewidth=2)

    plt.title(f'Neuron Activation Counts per Layer ({L1} and {L2})')
    plt.xlabel('Layer Index')
    plt.ylabel('Number of Neurons')
    plt.xticks(range(num_layers))
    plt.legend()
    plt.grid()

    # グラフの保存
    plt.savefig(f'/home/s2410121/proj_LA/activated_neuron/images/{folder}/activated_neuron_{L1}_{L2}.png')
    plt.close()

def visualize_neurons_with_line_plot_b(
                                    L1,
                                    L2,
                                    # main
                                    activated_neurons_L1,
                                    activated_neurons_L2,
                                    non_activated_neurons_L1,
                                    non_activated_neurons_L2,
                                    shared_neurons,
                                    specific_neurons_L1,
                                    specific_neurons_L2,
                                    non_activated_neurons_all,
                                    folder: str,
                                    ):
    # nums of all layers(LLaMA-3-8B)
    num_layers = 32

    # list for aggregating all types of neurons
    """ main """
    L2_counts = [0] * num_layers
    L1_counts = [0] * num_layers
    shared_counts = [0] * num_layers
    specific_L2_counts = [0] * num_layers
    specific_L1_counts = [0] * num_layers
    non_activated_L2_counts = [0] * num_layers  # 日本語の非活性化ニューロン
    non_activated_L1_counts = [0] * num_layers  # 英語の非活性化ニューロン
    non_activated_all_counts = [0] * num_layers  # 両言語の非活性化共通ニューロン

    # counting activate/non-activate counts
    for layer_idx in range(num_layers):
        # activated_neurons_L1 と activated_neurons_L2 から、各層のユニークなニューロン数を取得
        L1_neurons_set = set(activated_neurons_L1[layer_idx][1][:, 2])  # 第3列がニューロンインデックス
        L2_neurons_set = set(activated_neurons_L2[layer_idx][1][:, 2])

        L2_counts[layer_idx] = len(L2_neurons_set)  # nums for ja neurons
        L1_counts[layer_idx] = len(L1_neurons_set)  # nums for en neurons
        shared_counts[layer_idx] = len(set(shared_neurons[layer_idx][1]))  # shared_neurons
        specific_L2_counts[layer_idx] = len(set(specific_neurons_L2[layer_idx][1].flatten()))  # specific neurons for ja
        specific_L1_counts[layer_idx] = len(set(specific_neurons_L1[layer_idx][1].flatten()))  # specific neurons for en
        non_activated_L2_counts[layer_idx] = len(set(non_activated_neurons_L2[layer_idx][1].flatten()))  # non-activate ja
        non_activated_L1_counts[layer_idx] = len(set(non_activated_neurons_L1[layer_idx][1].flatten()))  # non-activate en
        non_activated_all_counts[layer_idx] = len(set(non_activated_neurons_all[layer_idx][1].flatten()))  # non-activate for both

    # plot
    plt.figure(figsize=(15, 10))
    plt.plot(range(num_layers), L2_counts, label=f'{L2} Activated Neurons', marker='o')
    plt.plot(range(num_layers), L1_counts, label=f'{L1} Activated Neurons', marker='o')
    plt.plot(range(num_layers), shared_counts, label='Shared Neurons', marker='o', linewidth=6)
    plt.plot(range(num_layers), specific_L2_counts, label=f'Specific to {L2}', marker='o')
    plt.plot(range(num_layers), specific_L1_counts, label=f'Specific to {L1}', marker='o')
    plt.plot(range(num_layers), non_activated_L2_counts, label=f'Non-Activated {L2} Neurons', marker='x', linestyle='--')
    plt.plot(range(num_layers), non_activated_L1_counts, label=f'Non-Activated {L1} Neurons', marker='x', linestyle='--')
    plt.plot(range(num_layers), non_activated_all_counts, label='Non-Activated Neurons (Both)', marker='s', linestyle='-.')

    plt.title(f'Neuron Activation Counts per Layer ({L1} and {L2})')
    plt.xlabel('Layer Index')
    plt.ylabel('Number of Neurons')
    plt.xticks(range(num_layers))
    plt.legend()
    plt.grid()

    # グラフの保存
    plt.savefig(f'/home/s2410121/proj_LA/activated_neuron/images/{folder}/activated_neuron_{L1}_{L2}.png')
    plt.close()

def visualize_neurons_with_line_plot_with_percent(
                                    L1,
                                    L2,
                                    activated_neurons_L1,
                                    activated_neurons_L2,
                                    non_activated_neurons_L1,
                                    non_activated_neurons_L2,
                                    shared_neurons,
                                    specific_neurons_L1,
                                    specific_neurons_L2,
                                    non_activated_neurons_all,
                                    folder: str,
                                    ):
    # nums of layers
    num_layers = 32

    # list for aggregating all types of neurons （with %）
    L2_counts = [0] * num_layers
    L1_counts = [0] * num_layers
    shared_counts = [0] * num_layers
    specific_L2_counts = [0] * num_layers
    specific_L1_counts = [0] * num_layers
    non_activated_L2_counts = [0] * num_layers  # 日本語の非活性化ニューロン
    non_activated_L1_counts = [0] * num_layers  # 英語の非活性化ニューロン
    non_activated_all_counts = [0] * num_layers  # 両言語の非活性化共通ニューロン

    # count activated/non-activated neurons per each layer
    for layer_idx in range(num_layers):
        # activated_neurons_L1 と activated_neurons_L2 から、各層のユニークなニューロン数を取得
        L1_neurons_set = set(activated_neurons_L1[layer_idx][1][:, 2])  # 第3列がニューロンインデックス
        L2_neurons_set = set(activated_neurons_L2[layer_idx][1][:, 2])

        L2_counts[layer_idx] = len(L2_neurons_set) / nums_of_neurons_llama3 * 100  # 日本語のニューロン数の割合
        L1_counts[layer_idx] = len(L1_neurons_set) / nums_of_neurons_llama3 * 100  # 英語のニューロン数の割合
        shared_counts[layer_idx] = len(set(shared_neurons[layer_idx][1].flatten())) / nums_of_neurons_llama3 * 100  # 共有ニューロン数の割合
        specific_L2_counts[layer_idx] = len(set(specific_neurons_L2[layer_idx][1].flatten())) / nums_of_neurons_llama3 * 100  # 日本語特有のニューロン数の割合
        specific_L1_counts[layer_idx] = len(set(specific_neurons_L1[layer_idx][1].flatten())) / nums_of_neurons_llama3 * 100  # 英語特有のニューロン数の割合
        # non_activated_L2_counts[layer_idx] = len(set(non_activated_neurons_L2[layer_idx][1].flatten())) / nums_of_neurons_llama3 * 100  # 日本語の非活性化ニューロン数の割合
        # non_activated_L1_counts[layer_idx] = len(set(non_activated_neurons_L1[layer_idx][1].flatten())) / nums_of_neurons_llama3 * 100  # 英語の非活性化ニューロン数の割合
        # non_activated_all_counts[layer_idx] = len(set(non_activated_neurons_all[layer_idx][1].flatten())) / nums_of_neurons_llama3 * 100  # 両言語の非活性化共通ニューロン数の割合

    # plot
    plt.figure(figsize=(15, 10))
    plt.plot(range(num_layers), L2_counts, label=f'{L2} Activated Neurons (%)', marker='o')
    plt.plot(range(num_layers), L1_counts, label=f'{L1} Activated Neurons (%)', marker='o')
    plt.plot(range(num_layers), shared_counts, label='Shared Neurons (%)', marker='o', linewidth=6)
    plt.plot(range(num_layers), specific_L2_counts, label=f'Specific to {L2} (%)', marker='o')
    plt.plot(range(num_layers), specific_L1_counts, label=f'Specific to {L1} (%)', marker='o')
    # plt.plot(range(num_layers), non_activated_L2_counts, label=f'Non-Activated {L2} Neurons (%)', marker='x', linestyle='--')
    # plt.plot(range(num_layers), non_activated_L1_counts, label=f'Non-Activated {L1} Neurons (%)', marker='x', linestyle='--')
    # plt.plot(range(num_layers), non_activated_all_counts, label='Non-Activated Neurons (Both) (%)', marker='s', linestyle='-.')

    plt.title(f'Neuron Activation Percentages per Layer ({L1} and {L2})')
    plt.xlabel('Layer Index')
    plt.ylabel('Percentage of Neurons (%)')
    plt.xticks(range(num_layers))
    plt.legend()
    plt.grid()

    # save figures
    plt.savefig(f'/home/s2410121/proj_LA/activated_neuron/images/{folder}/activated_neuron_{L1}_{L2}_percentage.png')
    plt.close()

# def visualize_neurons_with_line_plot_with_subplots(
#                                     L1,
#                                     L2,
#                                     activated_neurons_L1,
#                                     activated_neurons_L2,
#                                     non_activated_neurons_L1,
#                                     non_activated_neurons_L2,
#                                     shared_neurons,
#                                     specific_neurons_L1,
#                                     specific_neurons_L2,
#                                     non_activated_neurons_all
#                                     ):
#     # 層の数
#     num_layers = 32

#     # 各層のニューロンの活性化を集計するためのリスト
#     L2_counts = [0] * num_layers
#     L1_counts = [0] * num_layers
#     shared_counts = [0] * num_layers
#     specific_L2_counts = [0] * num_layers
#     specific_L1_counts = [0] * num_layers
#     non_activated_L2_counts = [0] * num_layers  # 日本語の非活性化ニューロン
#     non_activated_L1_counts = [0] * num_layers  # 英語の非活性化ニューロン
#     non_activated_all_counts = [0] * num_layers  # 両言語の非活性化共通ニューロン

#     # 各層におけるニューロンの活性化数をカウント
#     for layer_idx in range(num_layers):
#         # activated_neurons_L1 と activated_neurons_L2 から、各層のユニークなニューロン数を取得
#         L1_neurons_set = set(activated_neurons_L1[layer_idx][1][:, 2])  # 第3列がニューロンインデックス
#         L2_neurons_set = set(activated_neurons_L2[layer_idx][1][:, 2])

#         L2_counts[layer_idx] = len(L2_neurons_set)  # 日本語のニューロン数
#         L1_counts[layer_idx] = len(L1_neurons_set)  # 英語のニューロン数
#         shared_counts[layer_idx] = len(set(shared_neurons[layer_idx][1].flatten()))  # 共有ニューロン数
#         specific_L2_counts[layer_idx] = len(set(specific_neurons_L2[layer_idx][1].flatten()))  # 日本語特有のニューロン数
#         specific_L1_counts[layer_idx] = len(set(specific_neurons_L1[layer_idx][1].flatten()))  # 英語特有のニューロン数
#         non_activated_L2_counts[layer_idx] = len(set(non_activated_neurons_L2[layer_idx][1].flatten()))  # 日本語の非活性化ニューロン数
#         non_activated_L1_counts[layer_idx] = len(set(non_activated_neurons_L1[layer_idx][1].flatten()))  # 英語の非活性化ニューロン数
#         non_activated_all_counts[layer_idx] = len(set(non_activated_neurons_all[layer_idx][1].flatten()))  # 両言語の非活性化共通ニューロン数

#     # プロット
#     fig, axs = plt.subplots(2, 1, figsize=(15, 12))

#     # Activated Neuronsのプロット
#     axs[0].plot(range(num_layers), L2_counts, label=f'{L2} Activated Neurons', marker='o')
#     axs[0].plot(range(num_layers), L1_counts, label=f'{L1} Activated Neurons', marker='o')
#     axs[0].plot(range(num_layers), shared_counts, label='Shared Neurons', marker='o', linewidth=6)
#     axs[0].plot(range(num_layers), specific_L2_counts, label=f'Specific to {L2}', marker='o')
#     axs[0].plot(range(num_layers), specific_L1_counts, label=f'Specific to {L1}', marker='o')
#     axs[0].set_title(f'Activated Neuron Counts per Layer ({L1} and {L2})')
#     axs[0].set_xlabel('Layer Index')
#     axs[0].set_ylabel('Number of Neurons')
#     axs[0].legend()
#     axs[0].grid()

#     # Non-Activated Neuronsのプロット
#     axs[1].plot(range(num_layers), non_activated_L2_counts, label=f'Non-Activated {L2} Neurons', marker='x', linestyle='--')
#     axs[1].plot(range(num_layers), non_activated_L1_counts, label=f'Non-Activated {L1} Neurons', marker='x', linestyle='--')
#     axs[1].plot(range(num_layers), non_activated_all_counts, label='Non-Activated Neurons (Both)', marker='s', linestyle='-.')
#     axs[1].set_title(f'Non-Activated Neuron Counts per Layer ({L1} and {L2})')
#     axs[1].set_xlabel('Layer Index')
#     axs[1].set_ylabel('Number of Neurons')
#     axs[1].legend()
#     axs[1].grid()

#     # グラフの保存
#     plt.tight_layout()
#     plt.savefig(f'/home/s2410121/proj_LA/activated_neuron/images/activated_neuron_{L1}_{L2}.png')
#     plt.close()

if __name__ == "__main__":
    # visualize_neurons_with_line_plot(L1, "ja")
    print("visualization completed !")
