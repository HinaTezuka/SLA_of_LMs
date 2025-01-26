import os
import sys
# sys.path.append("/home/s2410121/proj_LA/activated_neuron")
import dill as pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import umap.umap_ as umap
from baukit import TraceDict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

def get_hidden_states(model, tokenizer, device, data, is_norm=False) -> list:
    """
    extract hidden states only for last tokens.
    is_norm: whether normalization of hidden states is required or not.

    return: np.array for input of sklearn classification models.
    """
    num_layers = 32
    # return
    input_for_sklearn_model = [[] for _ in range(num_layers)] # [layer_idx: 2000 pairs(translation or non translation)]

    for L1_txt, L2_txt in data:
        hidden_states = defaultdict(torch.Tensor)
        inputs_L1 = tokenizer(L1_txt, return_tensors="pt").to(device)
        inputs_L2 = tokenizer(L2_txt, return_tensors="pt").to(device)

        # get hidden_states
        with torch.no_grad():
            output_L1 = model(**inputs_L1, output_hidden_states=True)
            output_L2 = model(**inputs_L2, output_hidden_states=True)

        all_hidden_states_L1 = output_L1.hidden_states
        all_hidden_states_L2 = output_L2.hidden_states
        # get last token index.
        last_token_index_L1 = inputs_L1["input_ids"].shape[1] - 1
        last_token_index_L2 = inputs_L2["input_ids"].shape[1] - 1

        """ 各層の最後のトークンのhidden stateをリストに格納 + 正規化 """
        if is_norm:
            last_token_hidden_states_L1 = np.array([
                (layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy() /
                np.linalg.norm(layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy(), axis=-1, keepdims=True))
                # embedding層(0層目)は排除
                for layer_hidden_state in all_hidden_states_L1[1:]
            ])
            last_token_hidden_states_L2 = np.array([
                (layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy() /
                np.linalg.norm(layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy(), axis=-1, keepdims=True))
                # embedding層(0層目)は排除
                for layer_hidden_state in all_hidden_states_L2[1:]
            ])
        elif not is_norm:
            last_token_hidden_states_L1 = np.array([
                layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy()
                for layer_hidden_state in all_hidden_states_L1[1:]
                ])
            last_token_hidden_states_L2 = np.array([
                layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy()
                for layer_hidden_state in all_hidden_states_L2[1:]
                ])
        """ make features per a layer and save it to list. """
        for i in range(num_layers):
            # 1次元化
            feature_L1 = last_token_hidden_states_L1[i][0]
            feature_L2 = last_token_hidden_states_L2[i][0]
            # concatenate L1 and L2 features
            features_L1_and_L2 = np.concatenate([feature_L1, feature_L2]) # 4096 + 4096 -> 8192次元
            input_for_sklearn_model[i].append(features_L1_and_L2)

    return input_for_sklearn_model # shape: (num_layers, num_pairs, 8192)

""" func for editing activation values """
def edit_activation(output, layer, layer_idx_and_neuron_idx):
    """
    edit activation value of neurons(indexed layer_idx and neuron_idx)
    output: activation values
    layer: sth like 'model.layers.{layer_idx}.mlp.act_fn'
    layer_idx_and_neuron_idx: list of tuples like [(layer_idx, neuron_idx), ....]
    """
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if str(layer_idx) in layer:  # layer名にlayer_idxが含まれているか確認
            output[:, :, neuron_idx] *= 0  # 指定されたニューロンの活性化値をゼロに設定

    return output

def get_hidden_states_intervention(model, tokenizer, device, data, deactivation_list, is_norm=False) -> list:
    """
    extract hidden states only for last tokens, while deactivating expert neurons(high AP score neurons).
    is_norm: whether normalization of hidden states is required or not.

    return: np.array for input of sklearn classification models.
    """
    trace_layers = [f'model.layers.{layer}.mlp.act_fn' for layer, _ in deactivation_list]
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, deactivation_list)) as tr:
        return get_hidden_states(model, tokenizer, device, data, is_norm)

def save_as_pickle(file_path: str, target_dict) -> None:
    """
    Save a dictionary as a pickle file with improved safety.
    """
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temp_path = file_path + ".tmp"  # Temporary file for safe writing

    try:
        # Write to a temporary file
        with open(temp_path, "wb") as f:
            pickle.dump(target_dict, f)
        # Replace the original file with the temporary file
        os.replace(temp_path, file_path)
        print("pkl_file successfully saved.")
    except Exception as e:
        # Clean up temporary file if an error occurs
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e  # Re-raise the exception for further handling

def unfreeze_pickle(file_path: str):
    """
    Load a pickle file as a dictionary with error handling.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Error unpickling file {file_path}: {e}")

def plot_pca(features_label1, features_label0, L2):
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
        plt.title(f'PCA Layer_{layer_idx+1}')
        plt.legend()
        plt.grid(True)

        # 画像の保存
        if intervention == "no":
            output_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/hidden_state_classification/pca/{L2}/layer_{layer_idx+1}.png"
        elif intervention == "yes":
            output_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/hidden_state_classification/pca/{L2}/intervention/layer_{layer_idx+1}.png"
        elif intervention == "base":
            output_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/hidden_state_classification/pca/{L2}/intervention_baseline/layer_{layer_idx+1}.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

def plot_plsr(features_label1, features_label0, L2, intervention="no"):
    features_label1 = np.array(features_label1)
    features_label0 = np.array(features_label0)

    for layer_idx in range(32):
        # 2次元配列に変換 (flatten: 各データを1次元ベクトル化)
        features_label1_layer = features_label1[layer_idx, :, :]
        features_label0_layer = features_label0[layer_idx, :, :]

        # データを結合
        all_features = np.concatenate([features_label1_layer, features_label0_layer], axis=0)
        all_labels = np.concatenate([
            np.ones(features_label1_layer.shape[0]),  # Label 1: 1
            np.zeros(features_label0_layer.shape[0])  # Label 0: 0
        ])

        # # データのスケーリング
        # scaler = StandardScaler()
        # all_features_scaled = scaler.fit_transform(all_features)

        # PLSRを適用
        pls = PLSRegression(n_components=2)
        all_features_2d = pls.fit_transform(all_features, all_labels)[0]

        # 元のデータに分割
        features_label1_2d = all_features_2d[:features_label1_layer.shape[0]]
        features_label0_2d = all_features_2d[features_label1_layer.shape[0]:]

        # プロット
        plt.figure(figsize=(8, 6))
        plt.scatter(features_label1_2d[:, 0], features_label1_2d[:, 1], color='green', label='Label 1', alpha=0.7)
        plt.scatter(features_label0_2d[:, 0], features_label0_2d[:, 1], color='purple', label='Label 0', alpha=0.7)
        plt.xlabel('PLSR Dimension 1')
        plt.ylabel('PLSR Dimension 2')
        plt.title(f'PLSR Layer_{layer_idx+1}')
        plt.legend()
        plt.grid(True)

        # 画像の保存
        if intervention == "no":
            output_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/hidden_state_classification/plsr/{L2}/layer_{layer_idx+1}.png"
        elif intervention == "yes":
            output_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/hidden_state_classification/plsr/{L2}/intervention/layer_{layer_idx+1}.png"
        elif intervention == "base":
            output_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/hidden_state_classification/plsr/{L2}/intervention_baseline/layer_{layer_idx+1}.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

def plot_umap(features_label1, features_label0, L2, intervention="no"):
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
        plt.title(f'UMAP Layer_{layer_idx+1}')
        plt.legend()
        plt.grid(True)

        # 画像の保存
        if intervention == "no":
            output_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/hidden_state_classification/umap/{L2}/layer_{layer_idx+1}.png"
        elif intervention == "yes":
            output_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/hidden_state_classification/umap/{L2}/intervention/layer_{layer_idx+1}.png"           
        elif intervention == "base":
            output_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/hidden_state_classification/umap/{L2}/intervention_baseline/layer_{layer_idx+1}.png"
        plt.savefig(output_path, bbox_inches="tight")
