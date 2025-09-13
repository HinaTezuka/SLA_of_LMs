import os
import sys
sys.path.append('activated_neuron/new_neurons/transfer_neurons/txt_generation')
import pickle
import collections
import random
import json

import numpy as np
from evaluate import load

from generation_funcs import (
    polywrite,
)

model_names = ['meta-llama/Meta-Llama-3-8B', 'CohereForAI/aya-expanse-8b', 'mistralai/Mistral-7B-v0.3']
model_name_dict = {
    'meta-llama/Meta-Llama-3-8B': 'Llama3-8B', 
    'mistralai/Mistral-7B-v0.3': 'Mistral-7B',
    'CohereForAI/aya-expanse-8b': 'Aya-expanse-8B',
}
langs = ['ja', 'nl', 'ko', 'it']
langs_for_polywrite = {
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
    'it': 'ita_Latn',
    'nl': 'nld_Latn',
}
langs_for_gc = {
    'ja': 'ja-JP',
    'it': 'it',
    'nl': 'nl',
}

def distinct_n(texts, n=2):
    """
    texts: 文字列のリスト
    n: n-gram の長さ
    return: Distinct-n スコア（0〜1）
    """
    total_ngrams = 0
    unique_ngrams = set()
    
    for text in texts:
        tokens = text.strip().split()
        ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)
    
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0


""" normal """
for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    model_name_for_saving = model_name_dict[model_name]

    print()
    print(model_name)
    print()

    for lang in langs:
        path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/txt_generation/results/{model_name_for_saving}_{lang}.json'
        with open(path, 'r', encoding="utf-8") as f:
            file = json.load(f)

        texts = []
        for ex in file:
            sample_idx = ex['sample_idx']
            texts.append(ex['output'])
        
        # calc DistinctN (N = {1, 2})
        print(f'=========== {lang} ===========')
        print("Distinct-1:", distinct_n(texts, n=1))
        print("Distinct-2:", distinct_n(texts, n=2))