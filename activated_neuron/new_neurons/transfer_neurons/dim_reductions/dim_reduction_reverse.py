import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import umap.umap_ as umap
from sklearn.decomposition import PCA

from funcs import (
    monolingual_dataset_en,
    get_hidden_states_including_emb_layer_with_edit_activation,
    get_hidden_states_including_emb_layer,
    get_centroid_of_shared_space,
    save_as_pickle,
    unfreeze_pickle,
)

langs = ["ja", "nl", "ko", "it", "en"]
# LLaMA3-8B / Mistral-7B / Aya-expanse-8B.
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", 'CohereForAI/aya-expanse-8b']
# model_names = ["mistralai/Mistral-7B-v0.3", 'CohereForAI/aya-expanse-8b']
device = "cuda" if torch.cuda.is_available() else "cpu"
num_layers = 33

def plot_pca(model_type: str, features_L1: dict, features_L2: dict, features_L3: dict, features_L4: dict, features_L5: dict, is_reverse: bool):
    languages = ["Japanese", "Dutch", "Korean", "Italian", "English"]
    colors = ["red", "blue", "yellow", "orange", "green"]

    if is_reverse:
        start, end = 20+1, 32+1
    else:
        start, end = 0, 20+1
    
    for layer_idx in range(start, end):  # Embedding layer + 32 hidden layers

        f1 = np.array(features_L1[layer_idx])
        f2 = np.array(features_L2[layer_idx])
        f3 = np.array(features_L3[layer_idx])
        f4 = np.array(features_L4[layer_idx])
        f5 = np.array(features_L5[layer_idx])

        all_features = np.concatenate([f1, f2, f3, f4, f5], axis=0)
        pca = PCA(n_components=2, random_state=42)
        pca.fit(all_features)

        # 各言語のデータを同じPCA空間に投影
        f1_2d = pca.transform(f1)
        f2_2d = pca.transform(f2)
        f3_2d = pca.transform(f3)
        f4_2d = pca.transform(f4)
        f5_2d = pca.transform(f5)

        # plot.
        plt.figure(figsize=(15, 12))
        for feats, color, label in zip([f1_2d, f2_2d, f3_2d, f4_2d, f5_2d], colors, languages):
            plt.scatter(feats[:, 0], feats[:, 1], color=color, label=label, alpha=0.7)

        plt.xlabel('PCA Dimension 1', fontsize=20)
        plt.ylabel('PCA Dimension 2', fontsize=20)

        title = 'Emb Layer' if layer_idx == 0 else f'Layer {layer_idx}'
        file_name = 'emb_layer.png' if layer_idx == 0 else f'{layer_idx}.png'
        plt.title(title, fontsize=25)
        plt.legend(fontsize=15)
        plt.grid(True)

        # save as image.
        if is_reverse:
            output_dir = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/dim_reduction/{model_type}/reverse'
        else:
            output_dir = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/dim_reduction/{model_type}/type-1'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

if __name__ == '__main__':
    n_list = [100, 1000, 3000, 5000]
    score_type = 'cos_sim'
    path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/mkqa_q_sentence_data_ja_nl_ko_it_en.pkl'
    sentences_all_langs = unfreeze_pickle(path)
    is_reverse = True
    for model_name in model_names:
        model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        for L2 in langs:
            # prepare type-2 Transfer Neurons.
            if L2 != "en":
                if is_reverse: # type-2
                    save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
                    sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
                    sorted_neurons = [neuron for neuron in sorted_neurons if neuron[0] in [ _ for _ in range(20, 32)]]
                else: # type-1
                    save_path_sorted_neurons = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/qa/{L2}_sorted_neurons_type1.pkl"
                    sorted_neurons = unfreeze_pickle(save_path_sorted_neurons)
                sorted_neurons = sorted_neurons[:1000]

            sentences = sentences_all_langs[L2]
            if L2 == 'en':
                hidden_states = get_hidden_states_including_emb_layer(model, tokenizer, device, num_layers, sentences)
            else:
                hidden_states = get_hidden_states_including_emb_layer_with_edit_activation(model, tokenizer, device, sorted_neurons, num_layers, sentences)
            # c_hidden_states: {layer_idx: [hs_sample1, hs_sample2, ...]}

            # save centroids as pkl.
            if is_reverse:
                save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/{L2}.pkl"
            else:
                save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/{L2}_type1.pkl"
            save_as_pickle(save_path, hidden_states)

        """ dim_reduction and plot with PCA. """
        # ["ja", "nl", "ko", "it", "en"]
        if is_reverse:
            hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/ja.pkl")
            hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/nl.pkl")
            hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/ko.pkl")
            hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/it.pkl")
            hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/reverse/en.pkl")
        else:
            hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ja_type1.pkl")
            hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/nl_type1.pkl")
            hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ko_type1.pkl")
            hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/it_type1.pkl")
            hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/en_type1.pkl")
        # 
        plot_pca(model_type, hs_ja, hs_nl, hs_ko, hs_it, hs_en, is_reverse)
        # plot_umap(model_type, hs_ja, hs_nl, hs_ko, hs_it, hs_en)

        del model
        torch.cuda.empty_cache()