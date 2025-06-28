import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import defaultdict

from funcs import (
    save_as_pickle,
    unfreeze_pickle,
)

import numpy as np

def distinct_n(sentences, n=1):
    all_ngrams = set()
    total = 0
    for sent in sentences:
        tokens = sent.strip().split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngrams = list(ngrams)
        all_ngrams.update(ngrams)
        total += len(ngrams)
    return len(all_ngrams) / total if total > 0 else 0.0


model_path_dict = {
    'llama3': 'meta-llama/Meta-Llama-3-8B',
    'mistral': 'mistralai/Mistral-7B-v0.3',
    'aya': 'CohereForAI/aya-expanse-8b',
}
model_types = ['llama3', 'aya', 'mistral']
langs = ['ja', 'ko', 'fr']
is_baselines = [True, False]

for model_type in model_types:
    print(f'=============================== {model_type} ===============================')
    for L2 in langs:
        for is_baseline in is_baselines:
            save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{L2}.pkl' if not is_baseline else f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{L2}_baseline.pkl'
            sentences = unfreeze_pickle(save_path)

            distinct_1 = distinct_n(sentences, n=1)
            distinct_2 = distinct_n(sentences, n=2)

            if is_baseline:
                print(f'Lang: {L2}, Baseline deactivated: → Dist-1: {distinct_1:.4f}, Dist-2: {distinct_2:.4f}')
            elif not is_baseline:
                print(f'Lang: {L2}, Type-1 deactivated: → Dist-1: {distinct_1:.4f}, Dist-2: {distinct_2:.4f}')