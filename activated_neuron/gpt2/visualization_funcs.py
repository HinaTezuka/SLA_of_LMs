import os
import sys

import numpy as np
import matplotlib.pyplot as plt

nums_of_neurons_llama3 = 3072 # nums of all neurons (GPT2-small: MLP)

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
    num_layers = 12

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
    plt.savefig(f'/home/s2410121/proj_LA/activated_neuron/gpt2/images/{folder}/activated_neuron_{L1}_{L2}.png')
    plt.close()
