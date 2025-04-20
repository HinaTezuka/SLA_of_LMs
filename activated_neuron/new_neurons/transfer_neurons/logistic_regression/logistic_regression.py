import os
import sys
import pickle
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
model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b']
device = "cuda" if torch.cuda.is_available() else "cpu"
langs = ['ja', 'nl', 'ko', 'it']
L1 = "en" # L1 is fixed to english.

for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'

    for L2 in langs:
        """ tatoeba translation corpus """
        dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
        # # select first 2000 sentences.
        total_sentence_num = 2000 if L2 == "ko" else 5000
        num_sentences = 1000
        dataset = dataset.select(range(total_sentence_num))
        # tatoeba_data = []
        # for sentence_idx, item in enumerate(dataset):
        #     if sentence_idx == num_sentences: break
        #     # check if there are empty sentences.
        #     if item['translation'][L1] != '' and item['translation'][L2] != '':
        #         tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
        # tatoeba_data_len = len(tatoeba_data)

        # same semantics sentence pairs: test split.
        tatoeba_data = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_multi_test.pkl")

        """ non-translation pair for baseline. """
        random_data = []
        if L2 == "ko": # koreanはデータ数が足りない
            dataset2 = load_dataset("tatoeba", lang1=L1, lang2="ja", split="train").select(range(5000))
        for sentence_idx, item in enumerate(dataset):
            if sentence_idx == num_sentences: break
            if L2 == "ko" and dataset2['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
                random_data.append((dataset2["translation"][num_sentences+sentence_idx][L1], item["translation"][L2])) 
            elif L2 != "ko" and dataset['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
                random_data.append((dataset["translation"][num_sentences+sentence_idx][L1], item["translation"][L2]))
        
        """ extract hidden states """
        # shape: (num_layers, num_pairs, 8192) <- layerごとに回帰モデルをつくるため.
        features_label1 = get_hidden_states(model, tokenizer, device, tatoeba_data)
        features_label0 = get_hidden_states(model, tokenizer, device, random_data)
        
        """ train & test logistic regression model """
        # parameters
        num_layers = 32
        # num_sentences = 2000

        # define labels
        labels_label1 = np.ones((num_sentences)) # 対訳ペアのラベル(1)
        labels_label0 = np.zeros((num_sentences)) # 対訳ペアのラベル(0)

        # layerごとに回帰モデルをtrain & test.
        layer_scores = []

        # train & test
        for layer_idx in range(num_layers):
            # get features of each layer
            X_label1 = features_label1[layer_idx] # 対訳
            X_label0 = features_label0[layer_idx] # 非対訳

            # label1 と label0 の特徴量を結合 (shape: (num_pairs * 2, concatenated_hiddenstates(L1/L2)_dim)
            X = np.vstack([X_label1, X_label0])  # 4000ペアの特徴量(対訳+非対訳)
            y = np.hstack([labels_label1, labels_label0])  # 対応するラベル（1と0）

            # logistic regression model
            r_model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=10000, random_state=42)
            # cross validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            """ check if trainings were converged. """
            # fit model with one layer's data to check convergence
            # model.fit(X, y)
            # print(f"Layer {layer_idx}: Converged: {model.n_iter_}")
            # sys.exit()

            scoring = ['accuracy', 'precision', 'recall', 'f1']
            scores = cross_validate(r_model, X, y, cv=cv, scoring=scoring)

            # save scores for each layer
            layer_scores.append({
                'layer': layer_idx,
                'accuracy': scores['test_accuracy'],
                'precision': scores['test_precision'],
                'recall': scores['test_recall'],
                'f1': scores['test_f1'],
            })
        # show scores
        print(f"=================== {L2} ===================")
        for result in layer_scores:
            print(f"Layer {result['layer']}: test_accuracy = {np.mean(result['accuracy']):.4f} ± {np.std(result['accuracy']):.4f}")
            print(f"Layer {result['layer']}: test_f1 = {np.mean(result['f1']):.4f} ± {np.std(result['f1']):.4f}")

        """ save scores as pkl. """
        path = f'activated_neuron/new_neurons/pickles/transfer_neurons/logistic_regression/{model_type}/{L2}.pkl'
        save_as_pickle(path, layer_scores)
        print(f"pkl saved.: {L2}")
        unfreeze_pickle(path)
        print(f"successfully unfreezed: {L2}")
    
    # cache
    del model
    torch.cuda.empty_cache()