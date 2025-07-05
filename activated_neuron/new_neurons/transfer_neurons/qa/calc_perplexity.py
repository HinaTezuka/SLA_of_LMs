import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import defaultdict
import json

# from funcs import (
#     save_as_pickle,
#     unfreeze_pickle,
# )

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_perplexity(sentences, model, tokenizer):
    perplexities = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
    
    return perplexities

# def calculate_mean_log_prob(sentences, model, tokenizer):
#     log_probs = []

#     for sentence in sentences:
#         inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
#         input_ids = inputs.input_ids.to(device)

#         # 文が極端に短い場合はスキップ（トークン長1以下）
#         if input_ids.size(1) < 2:
#             continue

#         with torch.no_grad():
#             outputs = model(input_ids, labels=input_ids)
        
#         loss = outputs.loss  # これは「mean negative log likelihood」
#         if loss is not None and not torch.isnan(loss):
#             log_prob = -loss.item()  # 平均 log probability
#             log_probs.append(log_prob)

#     return log_probs

model_path_dict = {
    'llama3': 'meta-llama/Meta-Llama-3-8B',
    'mistral': 'mistralai/Mistral-7B-v0.3',
    'aya': 'CohereForAI/aya-expanse-8b',
}
# model_types = ['aya', 'llama3', 'mistral']
model_types = ['aya', 'llama3']
# model_types = ['llama3', 'aya']
langs = ['ja', 'nl', 'ko']
num_to_extract = 100

for model_type in model_types:
    tokenizer = AutoTokenizer.from_pretrained(model_path_dict[model_type])
    model = AutoModelForCausalLM.from_pretrained(model_path_dict[model_type]).to(device)
    model.eval()
    for L2 in langs:
        for deactivation_type in ['normal', 'type2', 'type1']:
            if deactivation_type == 'normal':
                save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/normal_ja.json'
            elif deactivation_type == 'type1':
                save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/{deactivation_type}_ja_{L2}.json' if model_type == 'llama3' else f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/{deactivation_type}_ja_{L2}_100.json'
            else: 
                save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/{deactivation_type}_ja_{L2}.json' if model_type == 'llama3' else f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/{deactivation_type}_ja_{L2}_100.json'
            with open(save_path, 'r', encoding='utf-8') as f:
                sentences = json.load(f)

            """ calc perplexity. """
            perplexities = calculate_perplexity(sentences, model, tokenizer)
            # perplexities = calculate_mean_log_prob(sentences, model, tokenizer)
            avg_perplexity = np.nanmean(perplexities)

            print(f'Model: {model_type}, Input Lang: {L2}, Deactivation Type: {deactivation_type}, Deactivation Lang: {L2} → Avg Perplexity: {avg_perplexity:.2f}')
            # print(f'Model: {model_type}, Lang: {L2}, Baseline: {is_baseline} → Avg Perplexity: {avg_perplexity:.2f}')

    del model
    torch.cuda.empty_cache() 