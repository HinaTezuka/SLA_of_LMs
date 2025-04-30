import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

model_types = ['llama3', 'mistral', 'aya']
langs = ['ja', 'nl', 'ko', 'it']
num_layers = 32

for model_type in model_types:
    acc_matrix = np.zeros((len(langs), num_layers))  # 行: 言語, 列: レイヤー

    for i, lang in enumerate(langs):
        path = f'activated_neuron/new_neurons/pickles/transfer_neurons/logistic_regression/{model_type}/{lang}.pkl'
        with open(path, 'rb') as f:
            layer_scores = pickle.load(f)
        for score in layer_scores:
            acc_matrix[i, score['layer']] = np.mean(score['accuracy'])

    # plot
    plt.figure(figsize=(15, 4))
    ax = sns.heatmap(
        acc_matrix,
        annot=True,
        fmt=".2f",
        cmap='Blues',
        xticklabels=range(1, num_layers + 1),
        yticklabels=[f"en-{l}" for l in langs],
        cbar_kws={'label': 'Test Accuracy'},
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 10},
        square=True,
    )

    title = 'LLaMA3-8B' if model_type == 'llama3' else 'Mistral-7B' if model_type == 'mistral' else 'Aya expanse-8B'
    ax.set_title(title, fontsize=20)
    if model_type == 'llama3':
        ax.set_xlabel("Layer Index", fontsize=15)
        ax.set_ylabel("Language Pair", fontsize=15)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)

    # ラベルを横向きに
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/logistic_regression/{model_type}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()