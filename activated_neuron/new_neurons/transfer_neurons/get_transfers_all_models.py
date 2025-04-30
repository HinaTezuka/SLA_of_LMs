import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    monolingual_dataset,
    compute_scores,
    compute_scores_optimized,
    sort_neurons_by_score,
    save_as_pickle,
    unfreeze_pickle,
)

# aya-expanse-8B.
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", "CohereForAI/aya-expanse-8b"]
device = "cuda" if torch.cuda.is_available() else "cpu"
langs = ["ja", "nl", "ko", "it", "en"]
score_types = ["cos_sim", "L2_dis"]
is_reverses = [True, False]

def get_sentences_qa(qa, L2: str, qa_num: int):
    l = []
    for i in range(len(qa['queries'])):
        if i == qa_num: break # 1000 questions.
        """ make prompt and input_ids. """
        q = qa['queries'][i][L2] # question
        a = qa['answers'][i][L2][0]['text'] # answer
        if q == '' or q == None or  a == '' or a == None:
            continue
        # ans_patterns = {
        # 'ja': '答え: ',
        # 'nl': 'Antwoord: ',
        # 'ko': '답변: ',
        # 'it': 'Risposta: ',
        # 'en': 'Answer: ',
        # }
        # prompt = f'{q}? {ans_patterns[L2]}'
        l.append(q)
    
    return l

qa = load_dataset('apple/mkqa')['train']
qa = qa.shuffle(seed=42)

for is_reverse in is_reverses:
    """ candidate neurons. """
    candidates = {}
    for layer_idx in range(32):
        for neuron_idx in range(14336):
            candidates.setdefault(layer_idx, []).append(neuron_idx)

    for model_name in model_names:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
        for is_reverse in is_reverses:
          for L2 in langs:
              if not is_reverse:
                  c_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/qa/c_qa_tran_en_{L2}.pkl"
              elif is_reverse:
                  c_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_{L2}_qa.pkl"
              centroids = unfreeze_pickle(c_path)
              monolingual_sentences = get_sentences_qa(qa, L2, 1000)
              for score_type in score_types:
                  # scores: {(layer_idx, neuron_idx): score, ....}
                  scores = compute_scores_optimized(model, tokenizer, device, monolingual_sentences, candidates, centroids, score_type) # for L2 only: reverse transfers.
                  # 降順
                  sorted_neurons, score_dict = sort_neurons_by_score(scores) # np
                  
                  # save as pkl.
                  if not is_reverse:
                    sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/qa/{L2}_sorted_neurons_type1.pkl"
                    score_dict_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/qa/{L2}_score_dict_type1.pkl"
                  else:
                    sorted_neurons_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/qa/{L2}_sorted_neurons.pkl"
                    score_dict_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/qa/{L2}_score_dict.pkl"
                  save_as_pickle(sorted_neurons_path, sorted_neurons)
                  save_as_pickle(score_dict_path, score_dict)
                  
                  del scores, sorted_neurons, score_dict
                  torch.cuda.empty_cache()
              print(f"saved scores for: {L2}.")
        
        del model
        torch.cuda.empty_cache()