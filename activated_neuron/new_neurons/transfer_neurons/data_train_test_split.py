import os
import sys
import pickle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    monolingual_dataset,
    monolingual_dataset_en,
    sentence_pairs_for_train_test_split,
    save_as_pickle,
    unfreeze_pickle,
)

num_sentences = 2000
split_num = 1000
langs = ["ja", "nl", "ko", "it"]

for L2 in langs:
    # monolingual data.
    monolingual_sentences = monolingual_dataset(L2, num_sentences)
    # multilingual data.
    multilingual_sentence_pairs = sentence_pairs_for_train_test_split(L2, num_sentences)

    mono_train = monolingual_sentences[:split_num]
    mono_test = monolingual_sentences[split_num:]
    multi_train = multilingual_sentence_pairs[:split_num]
    multi_test = multilingual_sentence_pairs[split_num:]

    # save as pkl.
    path_mono_train = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_mono_train.pkl"
    path_mono_test = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_mono_test.pkl"
    path_multi_train = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_multi_train.pkl"
    path_multi_test = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_multi_test.pkl"
    save_as_pickle(path_mono_train, mono_train)
    save_as_pickle(path_mono_test, mono_test)
    save_as_pickle(path_multi_train, multi_train)
    save_as_pickle(path_multi_test, multi_test)

""" data for en-only """
# en
en_sentences = monolingual_dataset_en(num_sentences)
en_train = en_sentences[:split_num]
en_test = en_sentences[split_num:]
path_en_train = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/en_mono_train.pkl"
path_en_test = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/en_mono_test.pkl"
save_as_pickle(path_en_train, en_train)
save_as_pickle(path_en_test, en_test)