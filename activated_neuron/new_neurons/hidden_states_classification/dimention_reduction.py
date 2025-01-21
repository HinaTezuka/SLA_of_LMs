import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/hidden_states_classification")
import dill as pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import umap.umap_ as umap
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

from funcs import (
    get_hidden_states,
    save_as_pickle,
    unfreeze_pickle,
)

""" extract hidden states (only for last token) and make inputs for the model. """

""" model configs """
# LLaMA-3(8B)
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    # "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    # "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    # "ko": "beomi/Llama-3-KoEn-8B", # ko
}
device = "cuda" if torch.cuda.is_available() else "cpu"

for L2, model_name in model_names.items():
    L1 = "en" # L1 is fixed to english.

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    
    """ tatoeba translation corpus """
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # select first 2000 sentences
    num_sentences = 2000
    dataset = dataset.select(range(num_sentences))
    tatoeba_data = []
    for item in dataset:
        # check if there are empty sentences.
        if item['translation'][L1] != '' and item['translation'][L2] != '':
            tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
    # tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]
    tatoeba_data_len = len(tatoeba_data)

    """
    baselineとして、対訳関係のない1文ずつのペアを作成
    (L1(en)はhttps://huggingface.co/datasets/agentlans/high-quality-english-sentences,
    L2はtatoebaの該当データを使用)
    """
    random_data = []
    # L1(en)
    en_base_ds = load_dataset("agentlans/high-quality-english-sentences")
    random_data_en = en_base_ds["train"][:num_sentences]
    en_base_ds_idx = 0
    for item in dataset:
        random_data.append((random_data_en["text"][en_base_ds_idx], item["translation"][L2]))
        en_base_ds_idx += 1
    
    """ extract hidden states """
    # shape: (num_layers, num_pairs, 8192) <- layerごとに回帰モデルをつくるため.
    features_label1 = get_hidden_states(model, tokenizer, device, tatoeba_data)
    features_label0 = get_hidden_states(model, tokenizer, device, random_data)
    
    def plot_pca(features_label1, features_label0):
        """
        PCAで次元削減し、結果をプロットする関数。

        Args:
            features_label1: ラベル1に対応する特徴量（リストまたはNumPy配列）。
            features_label0: ラベル0に対応する特徴量（リストまたはNumPy配列）。
        """
        # 入力をNumPy配列に変換
        features_label1 = np.array(features_label1)
        features_label0 = np.array(features_label0)

        for layer_idx in range(32):

            # 2次元配列に変換 (flatten: 各データを1次元ベクトル化)
            features_label1_layer = features_label1[layer_idx, : :]
            features_label0_layer = features_label0[layer_idx, :, :]

            """ PCA after concat """
            # # 2つのデータセットを結合
            # all_features = np.concatenate([features_label1_layer, features_label0_layer], axis=0)
            # print(all_features.shape)

            # pca = PCA(n_components=2)
            # all_features_2d = pca.fit_transform(all_features)

            # # 分割して取得
            # features_label1_2d = all_features_2d[:len(features_label1_layer)]
            # features_label0_2d = all_features_2d[len(features_label1_layer):]

            """ PCA sepalately """
            pca = PCA(n_components=2)
            features_label1_2d = pca.fit_transform(features_label1_layer)
            features_label0_2d = pca.fit_transform(features_label0_layer)

            # プロット
            plt.figure(figsize=(8, 6))
            plt.scatter(features_label1_2d[:, 0], features_label1_2d[:, 1], color='blue', label='Label 1', alpha=0.7)
            plt.scatter(features_label0_2d[:, 0], features_label0_2d[:, 1], color='red', label='Label 0', alpha=0.7)
            plt.xlabel('PCA Dimension 1')
            plt.ylabel('PCA Dimension 2')
            plt.title('PCA Projection of Features')
            plt.legend()
            plt.grid(True)

            # 画像の保存
            output_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/hidden_state_classification/pca/{L2}/layer_{layer_idx}.png"
            plt.savefig(output_path, bbox_inches="tight")
    
    def plot_umap(features_label1, features_label0):
        """
        UMAPで次元削減し、結果をプロットする関数。

        Args:
            features_label1: ラベル1に対応する特徴量（リストまたはNumPy配列）。
            features_label0: ラベル0に対応する特徴量（リストまたはNumPy配列）。
            output_path: プロット画像を保存するパス（文字列）。
        """
        # input list to np.array
        features_label1 = np.array(features_label1)
        features_label0 = np.array(features_label0)

        for layer_idx in range(32):

            # 2次元配列に変換 (flatten: 各データを1次元ベクトル化)
            features_label1_layer = features_label1[layer_idx, : :]
            features_label0_layer = features_label0[layer_idx, :, :]
            
            """ UMAP (after concat) """
            # # 2つのデータセットを結合
            # all_features = np.concatenate([features_label1_layer, features_label0_layer], axis=0)
            # print(all_features.shape)

            # # UMAPで2次元に削減
            # reducer = umap.UMAP(n_components=2, random_state=42)
            # all_features_2d = reducer.fit_transform(all_features)

            # # 分割して取得
            # features_label1_2d = all_features_2d[:len(features_label1_layer)]
            # features_label0_2d = all_features_2d[len(features_label1_layer):]

            """ UMAP (separately) """
            reducer = umap.UMAP(n_components=2, random_state=42)
            features_label1_2d = reducer.fit_transform(features_label1_layer)
            features_label0_2d = reducer.fit_transform(features_label0_layer)

            # プロット
            plt.figure(figsize=(8, 6))
            plt.scatter(features_label1_2d[:, 0], features_label1_2d[:, 1], color='blue', label='Label 1', alpha=0.7)
            plt.scatter(features_label0_2d[:, 0], features_label0_2d[:, 1], color='red', label='Label 0', alpha=0.7)
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.title('UMAP Projection of Features')
            plt.legend()
            plt.grid(True)

            # 画像の保存
            output_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/hidden_state_classification/umap/{L2}/layer_{layer_idx}.png"
            plt.savefig(output_path, bbox_inches="tight")

    # plot_pca(features_label1, features_label0)
    plot_umap(features_label1, features_label0)
