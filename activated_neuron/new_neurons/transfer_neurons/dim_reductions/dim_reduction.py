import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from funcs import (
    monolingual_dataset_en,
    get_hidden_states_including_emb_layer,
    get_centroid_of_shared_space,
    save_as_pickle,
    unfreeze_pickle,
)

langs = ["ja", "nl", "ko", "it", "en", "vi", "ru", "fr"]
# LLaMA3-8B / Mistral-7B / Aya-expanse-8B / Phi4-14B / Qwen3-8B.
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", 'CohereForAI/aya-expanse-8b', "microsoft/phi-4"]
model_names = ['Qwen/Qwen3-8B']
device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_pca(model_type: str, features_L1: dict, features_L2: dict, features_L3: dict, features_L4: dict, features_L5: dict, features_L6: dict, features_L7: dict, features_L8: dict):
    languages = ["Japanese", "Dutch", "Korean", "Italian", "English", "Vietnamese", "Russian", "French"]
    colors = ["red", "blue", "yellow", "orange", "green", "purple", "cyan", "brown"]
    
    num_layers = 33 if model_type != 'phi4' else 41
    for layer_idx in range(num_layers):  # Embedding layer + 32 hidden layers

        f1 = np.array(features_L1[layer_idx])
        f2 = np.array(features_L2[layer_idx])
        f3 = np.array(features_L3[layer_idx])
        f4 = np.array(features_L4[layer_idx])
        f5 = np.array(features_L5[layer_idx])
        f6 = np.array(features_L6[layer_idx])
        f7 = np.array(features_L7[layer_idx])
        f8 = np.array(features_L8[layer_idx])

        all_features = np.concatenate([f1, f2, f3, f4, f5, f6, f7, f8], axis=0)
        if model_type == 'phi4':
            scaler = StandardScaler()
            all_features = scaler.fit_transform(all_features)
        pca = PCA(n_components=2, random_state=42)
        pca.fit(all_features)

        # 各言語のデータを同じPCA空間に投影
        f1_2d = pca.transform(f1)
        f2_2d = pca.transform(f2)
        f3_2d = pca.transform(f3)
        f4_2d = pca.transform(f4)
        f5_2d = pca.transform(f5)
        f6_2d = pca.transform(f6)
        f7_2d = pca.transform(f7)
        f8_2d = pca.transform(f8)

        # plot.
        plt.rcParams["font.family"] = "DejaVu Serif"
        plt.figure(figsize=(12, 12))
        for feats, color, label in zip([f1_2d, f2_2d, f3_2d, f4_2d, f5_2d, f6_2d, f7_2d, f8_2d], colors, languages):
            plt.scatter(feats[:, 0], feats[:, 1], color=color, label=label, alpha=0.7)
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label=lang,
                markerfacecolor=col, markersize=30, alpha=0.7)
            for lang, col in zip(languages, colors)
        ]

        plt.xlabel('Principal Component 1', fontsize=40)
        plt.ylabel('Principal Component 2', fontsize=40)

        title = 'Emb Layer' if layer_idx == 0 else f'Layer {layer_idx}'
        file_name = 'emb_layer' if layer_idx == 0 else f'{layer_idx}'
        plt.title(title, fontsize=50)
        plt.legend(handles=legend_handles, fontsize=35)
        plt.grid(True)

        # save as image.
        output_dir = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/dim_reduction/{model_type}/all/{file_name}'
        with PdfPages(output_dir + '.pdf') as pdf:
            pdf.savefig(bbox_inches='tight', pad_inches=0.01)
            plt.close()

if __name__ == '__main__':
    path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/mkqa_q_sentence_data_ja_nl_ko_it_en_vi_ru_fr.pkl'
    sentences_all_langs = unfreeze_pickle(path)
    path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/mkqa_q_sentence_data_ja_nl_ko_it_en.pkl'
    sentences_all_langs2 = unfreeze_pickle(path)

    for model_name in model_names:
        model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya' if 'aya' in model_name else 'phi4' if 'phi' in model_name else 'qwen'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type == 'phi4':
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        num_layers = model.config.num_hidden_layers

        for L2 in langs:
            sentences = sentences_all_langs[L2]
            hidden_states = get_hidden_states_including_emb_layer(model, tokenizer, device, num_layers, sentences)
            # hidden_states: {layer_idx: [hs_sample1, hs_sample2, ...]}

            # save as pkl
            save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/{L2}.pkl"
            save_as_pickle(save_path, hidden_states)

        """ dim_reduction and plot with PCA. """
        # ["ja", "nl", "ko", "it", "en", "vi", "ru", "fr"]
        hs_ja = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ja.pkl")
        hs_nl = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/nl.pkl")
        hs_ko = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ko.pkl")
        hs_it = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/it.pkl")
        hs_en = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/en.pkl")
        hs_vi = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/vi.pkl")
        hs_ru = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/ru.pkl")
        hs_fr = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/hidden_states/fr.pkl")
        # 
        plot_pca(model_type, hs_ja, hs_nl, hs_ko, hs_it, hs_en, hs_vi, hs_ru, hs_fr)

        # del model
        torch.cuda.empty_cache()