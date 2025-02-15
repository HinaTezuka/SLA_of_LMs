"""
making multilingual data for langage specific neurons.
"""
import sys
import dill as pickle

from datasets import load_dataset

""" tatoeba translation corpus for lang-specific neuron detection. """
num_sentences = 500
langs = ["ja", "nl", "ko", "it", "en"]
L1 = "en"

tatoeba_data = []
for lang in langs:
    if lang == "en":
        dataset = load_dataset("tatoeba", lang1="en", lang2="ja", split="train")
        dataset = dataset.select(range(num_sentences)) 
        for item in dataset:
            if item['translation'][lang] != '':
                tatoeba_data.append(item['translation'][lang])
        continue
    dataset = load_dataset("tatoeba", lang1="en", lang2=lang, split="train")
    dataset = dataset.select(range(num_sentences))
    for sentence_idx, item in enumerate(dataset):
        if item['translation'][lang] != '':
            tatoeba_data.append(item['translation'][lang])



