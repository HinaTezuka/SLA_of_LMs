import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/new_neurons/hidden_states_classification/gpt2")
import dill as pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
""" 
層化分割交差検証（Stratified fold cross-validation): 各分割内でのクラスの比率が全体の比率と同じになるように分割(ラベル分布を維持)
"""
from funcs import (
    get_hidden_states,
    save_as_pickle,
    unfreeze_pickle,
)

""" extract hidden states (only for last token) and make inputs for the model. """

""" model configs """
# GPT-2
model_names = {
    # "base": "openai-community/gpt2",
    "ja": "rinna/japanese-gpt2-small", # ja
    # "de": "ml6team/gpt2-small-german-finetune-oscar", # ger
    "nl": "GroNLP/gpt2-small-dutch", # du
    "it": "GroNLP/gpt2-small-italian", # ita
    "fr": "dbddv01/gpt2-french-small", # fre
    "ko": "skt/kogpt2-base-v2", # ko
    "es": "datificate/gpt2-small-spanish" # spa
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
    features_label1 = get_hidden_states(model, tokenizer, device, tatoeba_data, is_norm=True)
    features_label0 = get_hidden_states(model, tokenizer, device, random_data, is_norm=True)
    
    """ train & test logistic regression model """
    # parameters
    num_layers = 12
    num_pairs = 2000

    # define labels
    labels_label1 = np.ones((num_pairs)) # 対訳ペアのラベル(1)
    labels_label0 = np.zeros((num_pairs)) # 対訳ペアのラベル(0)

    # layerごとに回帰モデルをtrain & test.
    layer_scores = []

    # train & test
    for layer_idx in range(num_layers):
        # get features of each layer
        X_label1 = features_label1[layer_idx] # 対訳
        X_label0 = features_label0[layer_idx] # 非対訳

        # label1 と label0 の特徴量を結合 (shape: (num_pairs * 2, concatenated_hiddenstates(L1/L2)_dim)
        X = np.vstack([X_label1, X_label0])  # 4000ペアの特徴量
        y = np.hstack([labels_label1, labels_label0])  # 対応するラベル（1と0）

        # logistic regression model
        model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
        # cross validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        """ check if training is going to be converge. """
        # fit model with one layer's data to check convergence
        # model.fit(X, y)
        # print(f"Layer {layer_idx}: Converged: {model.n_iter_}") # ja:0層目は17回で収束
        # sys.exit()

        scoring = ['accuracy', 'precision', 'recall', 'f1']
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

        # save scores for each layer
        layer_scores.append({
            'layer': layer_idx,
            'accuracy': scores['test_accuracy'],
            'precision': scores['test_precision'],
            'recall': scores['test_recall'],
            'f1': scores['test_f1'],
        })
    # show scores
    for result in layer_scores:
        print(f"Layer {result['layer']}: test_accuracy = {np.mean(result['accuracy']):.4f} ± {np.std(result['accuracy']):.4f}")
        print(f"Layer {result['layer']}: test_f1 = {np.mean(result['f1']):.4f} ± {np.std(result['f1']):.4f}")

    """ save scores as pkl. """
    path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/pickles/logistic_regression/en_{L2}.pkl"
    save_as_pickle(path, layer_scores)
    print(f"pkl saved.: {L2}")
    unfreeze_pickle(path)
    print(f"successfully unfreezed: {L2}")