import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")
import dill as pickle
from collections import defaultdict

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from neuron_deactivation_funcs import (
    take_similarities_with_edit_activation,
    plot_hist,
)

if __name__ == "__main__":

    # L1 = english
    L1 = "en"
    """ model configs """
    # LLaMA-3
    model_names = {
        # "base": "meta-llama/Meta-Llama-3-8B"
        "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
        # "de": "DiscoResearch/Llama3-German-8B", # ger
        "nl": "ReBatch/Llama-3-8B-dutch", # du
        "it": "DeepMount00/Llama-3-8b-Ita", # ita
        "ko": "beomi/Llama-3-KoEn-8B", # ko
    }

    for L2, model_name in model_names.items():
        """ load pkl_file(act_sum_dict) """
        pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/pickles/act_sum/tatoeba_0_th/act_sum_dict/act_sum_dict_en_{L2}_tatoeba_0_th.pkl"
        with open(pkl_file_path, "rb") as f:
            act_sum_dict = pickle.load(f)
        print("unfolded pickle: act_sum_dict")

        # それぞれのneuronsの発火値の合計（dict)を取得
        act_sum_shared = act_sum_dict["shared"] # 非対訳ペアに発火しているshared neuronsも含む

        """ load pkl_file(act_freq_base_dict): 非対訳ペアに発火している shared neruons <- 対訳ペアのみに対して発火しているshared neuronsをとるため """
        pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/pickles/base/act_freq/tatoeba_0_th/act_freq_dict_en_{L2}_tatoeba_0_th.pkl"
        with open(pkl_file_path, "rb") as f:
            act_freq_base_dict = pickle.load(f)
        print("unfolded pickle: act_freq_base_dict")

        # shared neuronsのうち、非対訳ペアに対して発火したneuronsを取得 <- dict: {layer_idx: neuron_idx}
        freq_base_shared = act_freq_base_dict["shared_neurons"]

        """ shared_ONLY_dictをロード（同じ意味表現にのみ発火しているshared neurons） """
        pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/pickles/shared_same_semantics/shared_ONLY_dict_en_{L2}_tatoeba_0_th.pkl"
        with open(pkl_file_path, "rb") as f:
            act_sum_shared = pickle.load(f)
        print("unfolded pickle: act_same_semantics")

        count_shared_ONLY = 0
        for layer_idx in act_sum_shared.keys():
            count_shared_ONLY += len(act_sum_shared[layer_idx])

        """
        list[(layer_idx, neuron_idx), ...] <= 介入実験用
        listはact_sumを軸に降順にソート
        同じ意味表現（対訳ペア）のみにたいして発火しているshared neurons
        """
        shared_same_semantics = [] # shared neurons [(layer_idx, neuron_idx), ...]
        for layer_idx, neurons in act_sum_shared.items():
            for neuron_idx in neurons.keys():
                shared_same_semantics.append((layer_idx, neuron_idx))
        shared_same_semantics = sorted(shared_same_semantics, key=lambda x: act_sum_shared[x[0]][x[1]], reverse=True)

        """
        非対訳ペアにたいして発火しているshared neurons
        """
        non_translation_shared = []
        for layer_idx, neurons in freq_base_shared.items():
            for neuron_idx in neurons.keys():
                non_translation_shared.append((layer_idx, neuron_idx))
        non_translation_shared = sorted(non_translation_shared, key=lambda x: freq_base_shared[x[0]][x[1]], reverse=True)

        """ どのくらい介入するか(n) """
        intervention_num = count_shared_ONLY
        # intervention_num = 20000
        shared_same_semantics = shared_same_semantics[:intervention_num]
        non_translation_shared = non_translation_shared[:intervention_num]

        """ tatoeba translation corpus """
        dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
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
        baseとして、対訳関係のない1文ずつのペアを作成
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

        """ device configs """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = model_names[L2]
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        """ deactivate shared_neurons(same semantics) """
        similarities_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, shared_same_semantics, tatoeba_data)
        similarities_non_same_semantics = take_similarities_with_edit_activation(model, tokenizer, device, shared_same_semantics, random_data)
        final_results_same_semantics = defaultdict(float)
        final_results_non_same_semantics = defaultdict(float)
        for layer_idx in range(32): # ３２ layers
            final_results_same_semantics[layer_idx] = np.array(similarities_same_semantics[layer_idx]).mean()
            final_results_non_same_semantics[layer_idx] = np.array(similarities_non_same_semantics[layer_idx]).mean()
        plot_hist(final_results_same_semantics, final_results_non_same_semantics, L2, "same_semantics")
        # sys.exit()
        """ deactivate shared_neurons(non same semantics) """
        similarities_same_semantics_a = take_similarities_with_edit_activation(model, tokenizer, device, non_translation_shared, tatoeba_data)
        similarities_non_same_semantics_a = take_similarities_with_edit_activation(model, tokenizer, device, non_translation_shared, random_data)
        final_results_same_semantics_a = defaultdict(float)
        final_results_non_same_semantics_a = defaultdict(float)
        for layer_idx in range(32):
            final_results_same_semantics_a[layer_idx] = np.array(similarities_same_semantics_a[layer_idx]).mean()
            final_results_non_same_semantics_a[layer_idx] = np.array(similarities_non_same_semantics_a[layer_idx]).mean()
        plot_hist(final_results_same_semantics_a, final_results_non_same_semantics_a, L2, "non_same_semantics")

        # delete some cache
        del model
        torch.cuda.empty_cache()
