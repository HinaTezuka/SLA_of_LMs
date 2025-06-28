import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import defaultdict

from funcs import (
    save_as_pickle,
    unfreeze_pickle,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
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
model_types = ['aya', 'llama3', 'mistral']
langs = ['ja', 'ko', 'fr']
is_baselines = [True, False]
num_to_extract = 100

for model_type in model_types:
    model_path = model_path_dict[model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    for L2 in langs:
        for is_baseline in is_baselines:
            save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{L2}.pkl' if not is_baseline else f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{L2}_baseline.pkl'
            sentences = unfreeze_pickle(save_path)

            """ calc perplexity. """

            perplexities = calculate_perplexity(sentences, model, tokenizer)
            # perplexities = calculate_mean_log_prob(sentences, model, tokenizer)
            avg_perplexity = np.mean(perplexities)

            print(f'Model: {model_type}, Lang: {L2}, Baseline: {is_baseline} → Avg Perplexity: {avg_perplexity:.2f}')
            # print(f'Model: {model_type}, Lang: {L2}, Baseline: {is_baseline} → Avg Perplexity: {avg_perplexity:.2f}')

    del model
    torch.cuda.empty_cache() 