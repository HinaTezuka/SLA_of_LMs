import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from funcs import unfreeze_pickle
from bert_score import score as bert_score

langs = ['ja', 'ko', 'fr']
model_types = ['aya', 'llama3', 'mistral']
is_baselines = [True, False]

for model_type in model_types:
    for lang in langs:
        # Load baseline and not-baseline sentence lists
        baseline_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{lang}_baseline.pkl'
        not_baseline_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{lang}.pkl'

        baseline_sentences = unfreeze_pickle(baseline_path)
        not_baseline_sentences = unfreeze_pickle(not_baseline_path)

        # BERTScore
        P, R, F1 = bert_score(not_baseline_sentences, baseline_sentences, lang=lang, verbose=False)
        avg_f1 = F1.mean().item()

        print(f'Model: {model_type}, Lang: {lang} → Avg BERTScore F1 (type-1 deactivated vs baseline): {avg_f1:.4f}')