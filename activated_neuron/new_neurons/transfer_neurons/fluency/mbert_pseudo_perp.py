import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from funcs import unfreeze_pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mBERT
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
model.eval()

def pseudo_perplexity(sentence):
    tokens = tokenizer.tokenize(sentence)
    # if len(tokens) < 2:
    #     return None  # スキップ
    total_log_prob = 0.0
    count = 0
    for i in range(len(tokens)):
        masked_tokens = tokens.copy()
        masked_tokens[i] = tokenizer.mask_token
        input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + masked_tokens + [tokenizer.sep_token])
        input_tensor = torch.tensor([input_ids]).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits

        mask_index = input_ids.index(tokenizer.mask_token_id)
        true_token_id = tokenizer.convert_tokens_to_ids([tokens[i]])[0]
        log_probs = F.log_softmax(logits[0, mask_index], dim=-1)
        token_log_prob = log_probs[true_token_id].item()

        total_log_prob += token_log_prob
        count += 1

    return np.exp(-total_log_prob / count) if count > 0 else None


langs = ['ja', 'ko', 'fr']
model_types = ['aya', 'llama3', 'mistral']
model_types = ['llama3', 'aya', 'mistral']

for model_type in model_types:
    for lang in langs:
        baseline_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{lang}_baseline.pkl'
        not_baseline_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{lang}.pkl'

        baseline_sentences = unfreeze_pickle(baseline_path)
        not_baseline_sentences = unfreeze_pickle(not_baseline_path)

        # Compute pseudo-perplexity
        baseline_scores = [pseudo_perplexity(s) for s in baseline_sentences]
        not_baseline_scores = [pseudo_perplexity(s) for s in not_baseline_sentences]

        # Drop None (e.g., very short or invalid sentences)
        baseline_scores = [s for s in baseline_scores if s is not None]
        not_baseline_scores = [s for s in not_baseline_scores if s is not None]

        baseline_avg = np.mean(baseline_scores)
        not_baseline_avg = np.mean(not_baseline_scores)

        print(f'Model: {model_type}, Lang: {lang} → Baseline Pseudo-PPL: {baseline_avg:.2f}, Not-Baseline Pseudo-PPL: {not_baseline_avg:.2f}')