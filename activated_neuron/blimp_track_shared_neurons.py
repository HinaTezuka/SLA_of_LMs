"""
BLiMPの各項目でshared neuronsがどれくらい使われているかを確認
"""
import sys
import pickle

from collections import defaultdict

import numpy as np
import torch
import transformers

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from neuron_detection_funcs import act_llama3, get_out_llama3, track_neurons_with_text_data

# LLaMA-3
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    # "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    # "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    # "ko": "beomi/Llama-3-KoEn-8B", # ko
}

model = AutoModelForCausalLM.from_pretrained(model_names["it"]).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_names["it"])

def track_neurons_with_text_blimp(model, model_name, tokenizer, blimp_sentence, active_THRESHOLD=0):
    # Initialize lists for tracking neurons
    activated_neurons = []
    non_activated_neurons = []
    shared_neurons = []
    specific_neurons = []

    # Initialize sets for visualization
    num_layers = 32 if model_name == "llama" else 12

    # Track neurons with blimp sentence
    # sentence
    sentence = example["sentence_good"] if i == 0 else example["sentence_bad"]
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to("cuda")
    token_len = len(input_ids[0])
    mlp_activation = act_llama3(model, input_ids)

    for layer_idx in range(len(mlp_activation)):
        # Activated neurons
        activated_neurons_layer = torch.nonzero(mlp_activation[layer_idx] > active_THRESHOLD).cpu().numpy()
        # 最後のトークンだけ考慮する
        activated_neurons_layer = activated_neurons_layer[activated_neurons_layer[:, 1] == token_len - 1]
        activated_neurons.append((layer_idx, activated_neurons_layer))

    return activated_neurons

""" 重複数カウント用の辞書を作成 """
def make_dict(activated_neurons):
    return_dict = defaultdict(list)
    for layer_num in range(len(activated_neurons)):
        return_dict[layer_num] = np.array(activated_neurons[layer_num][1][:, 2])
    return return_dict

""" shared neuronsとの重複数をカウント """
def count_shared_neurons_blimp(activated_neurons_dict, shared_neurons_dict):
    return_dict = defaultdict(list)
    # 層ごとの重複数を保存
    counts = []
    for layer_idx in range(len(activated_neurons_dict)):
        # 該当のlayer_idxのshared_neuronsをnp.arrayに
        shared_neurons = np.array(list(shared_neurons_dict[layer_idx]))
        return_dict[layer_idx] = np.intersect1d(activated_neurons_dict[layer_idx], shared_neurons)
        counts.append(len(return_dict[layer_idx]))

    return return_dict, np.array(counts).mean()


if __name__ == "__main__":

    """ shared neuronsのロード(pickle file) """
    with open(f"/home/s2410121/proj_LA/activated_neuron/pickles/shared_neurons_en_it_tatoeba_01_th.pkl", "rb") as f:
        shared_neurons_dict = pickle.load(f)

    # BLiMPの評価項目リスト
    configs_ja = ["npi_present_1", "distractor_agreement_relative_clause", "irregular_past_participle_adjectives", "only_npi_licensor_present", "existential_there_quantifiers_2"]
    configs_it = ["existential_there_quantifiers_2", "matrix_question_npi_licensor_present", "npi_present_2", "irregular_past_participle_adjectives"]

    # 重複数を記録しておくリスト(各入力文における、各層の重複数をカウントし、平均したもの)
    track_duplication = [] # 32
    track_duplication_sentence_good = []
    track_duplication_sentence_bad = []
    # 各入力文における、各層のshared_neuronsの重複を保存するdict
    track_dict = defaultdict(dict)
    track_dict_sentence_good = defaultdict(dict)
    track_dict_sentence_bad = defaultdict(dict)

    for config in configs_it:
        blimp = load_dataset("blimp", config)
        sentence_cnt = 0
        for example in blimp["train"]:
            # sentence_good/bad判別用のflag
            is_sentence_good = False
            for i in range(2):
                is_sentence_good = True if i == 0 else False
                sentence = example["sentence_good"] if i == 0 else example["sentence_bad"]
                # get activated_neurons
                activated_neurons = track_neurons_with_text_blimp(model, "llama", tokenizer, sentence, 0.1)
                # get dict_version: {layer_idx: np.array(activated_neurons)}
                activated_neurons_dict = make_dict(activated_neurons)
                # それぞれの層におけるshared_neurons(shared_neurons_duplicate), 各層のshared_neuronsの重複の平均値(shared_counts_mean)を取得
                shared_neurons_duplicate, shared_counts_mean = count_shared_neurons_blimp(activated_neurons_dict, shared_neurons_dict)
                # track_dictに保存
                if is_sentence_good:
                    track_dict["sentence_good"][sentence_cnt] = shared_neurons_duplicate
                    track_duplication_sentence_good.append(shared_counts_mean)
                else:
                    track_dict["sentence_bad"][sentence_cnt] = shared_neurons_duplicate
                    track_duplication_sentence_bad.append(shared_counts_mean)
                # track_duplicationにmeanを保存
                track_duplication.append(shared_counts_mean)
            sentence_cnt += 1

        print(f"{config}: {int(np.array(track_duplication).mean())}")
        print(f"{config}(good): {int(np.array(track_duplication_sentence_good).mean())}")
        print(f"{config}(bad): {int(np.array(track_duplication_sentence_bad).mean())}")

