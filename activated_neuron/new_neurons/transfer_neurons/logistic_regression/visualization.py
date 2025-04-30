import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from matplotlib import gridspec

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

    if model_type == 'aya':
        fig = plt.figure(figsize=(15, 6.2))  # 高さ調整
        gs = gridspec.GridSpec(2, 1, height_ratios=[25, 1], hspace=0.01)  # ここを最小に
        ax = fig.add_subplot(gs[0])
        cbar_ax = fig.add_subplot(gs[1])
    else:
        plt.figure(figsize=(15, 6))
        ax = plt.gca()
        cbar_ax = None

    sns.heatmap(
        acc_matrix,
        annot=True,
        fmt=".2f",
        cmap='Blues',
        xticklabels=range(1, num_layers + 1),
        yticklabels=[f"en-{l}" for l in langs] if model_type == 'aya' else [],
        cbar=(model_type == 'aya'),
        cbar_ax=cbar_ax if model_type == 'aya' else None,
        cbar_kws={'orientation': 'horizontal', 'label': 'Test Accuracy'} if model_type == 'aya' else None,
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 10},
        square=True,
        ax=ax
    )

    title = 'LLaMA3-8B' if model_type == 'llama3' else 'Mistral-7B' if model_type == 'mistral' else 'Aya expanse-8B'
    ax.set_title(title, fontsize=20)

    if model_type == 'aya':
        ax.set_xlabel("Layer Index", fontsize=15)
        ax.set_ylabel("Language Pair", fontsize=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        fig.subplots_adjust(top=0.94, bottom=0.13, left=0.08, right=0.98)  # 上下余白を少し狭める
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.tight_layout()

    save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/logistic_regression/{model_type}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()