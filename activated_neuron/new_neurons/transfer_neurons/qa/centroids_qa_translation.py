"""
calc centroid.
centroid: center point of Language-Agnostic Space.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from collections import defaultdict

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    monolingual_dataset_en,
    get_hidden_states_en_only,
    get_centroid_of_shared_space,
    save_as_pickle,
    unfreeze_pickle,
)

""" """
def get_sentences_qa(qa, qa_num: int, L1: str, L2: str):
    prompts = []
    for i in range(len(qa['queries'])):
        if i == qa_num: break # 1000 questions.
        """ make prompt and input_ids. """
        q_L1 = qa['queries'][i][L1] # question
        q_L2 = qa['queries'][i][L2]
        ans_patterns = {
        'ja': '答え: ',
        'nl': 'Antwoord: ',
        'ko': '답변: ',
        'it': 'Risposta: ',
        'en': 'Answer: ',
        }
        prompt_L1 = f'{q_L1} {ans_patterns[L1]}'
        prompt_L2 = f'{q_L2} {ans_patterns[L2]}'
        prompts.append((prompt_L1, prompt_L2))
    
    return prompts

def get_c_tran(model, tokenizer, device, num_layers, data: list) -> list:
    """
    """
    # { layer_idx: [c_1, c_2, ...]} c_1: (last token)centroid of text1 (en-L2).
    c_hidden_states = defaultdict(list)

    for text1, text2 in data:
        inputs1 = tokenizer(text1, return_tensors="pt").to(device) # english text
        inputs2 = tokenizer(text2, return_tensors="pt").to(device) # L2 text

        # get hidden_states
        with torch.no_grad():
            output1 = model(**inputs1, output_hidden_states=True)
            output2 = model(**inputs2, output_hidden_states=True)

        all_hidden_states1 = output1.hidden_states[1:] # remove embedding layer
        all_hidden_states2 = output2.hidden_states[1:]
        last_token_index1 = inputs1["input_ids"].shape[1] - 1
        last_token_index2 = inputs2["input_ids"].shape[1] - 1

        """  """
        for layer_idx in range(num_layers):
            hs1 = all_hidden_states1[layer_idx][:, last_token_index1, :].squeeze().detach().cpu().numpy()
            hs2 = all_hidden_states2[layer_idx][:, last_token_index2, :].squeeze().detach().cpu().numpy()
            # save mean of (en_ht, L2_ht). <- estimated shared point in shared semantic space.
            c = np.stack([hs1, hs2])
            c = np.mean(c, axis=0)
            c_hidden_states[layer_idx].append(c)

    return get_centroid_of_shared_space(dict(c_hidden_states))

def get_centroid_of_shared_space(hidden_states: dict):
    centroids = [] # [c1, c2, ] len = layer_num(32layers: 0-31)
    for layer_idx, c in hidden_states.items():
        final_c = np.mean(c, axis=0) # calc mean of c(shared point per text) of all text.
        centroids.append(final_c)
    return centroids

langs = ["ja", "nl", "ko", "it", "en"]
# LLaMA3-8B / Mistral-7B
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", 'CohereForAI/aya-expanse-8b']
device = "cuda" if torch.cuda.is_available() else "cpu"
num_layers = 32
pair_patterns = {
    'en': [('en', 'ja'), ('en', 'nl'), ('en', 'ko'), ('en', 'it')],
    'ja': [('ja', 'nl'), ('ja', 'ko'), ('ja', 'it')],
    'nl': [('nl', 'ja'), ('nl', 'ko'), ('nl', 'it')],
    'ko': [('ko', 'ja'), ('ko', 'nl'), ('ko', 'it')],
    'it': [('it', 'ja'), ('it', 'nl'), ('it', 'ko')],
}

""" """

# load qa data and shuffle.
qa_num = 1000
qa = load_dataset('apple/mkqa')['train']
qa = qa.shuffle(seed=42)

# centroids of english texts.
centroids = {} # { L2: [shared_centroids(en-L2)_1, ...} <- len(values) = 32(layer_num).


for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    c_dict = {}
    for L2 in langs:

        # hidden_states for each L2.
        # mono L2 sentences.
        pair_pattern = pair_patterns[L2]
        for pair in pair_pattern:
            pair_L1, pair_L2 = pair[0], pair[1]
            sentences = get_sentences_qa(qa, qa_num, pair_L1, pair_L2) # [(sentence_L1, sentence_L2), ...]
            # # save qa sentences.
            # save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/sentences_tran_{pair_L1}_{pair_L2}.pkl'
            # save_as_pickle(save_path, sentences)
            
            # get centroids of hidden_states(of en-L2 sentence pairs).
            c_hidden_states = get_c_tran(model, tokenizer, device, num_layers, sentences)

            # save centroids as pkl.
            save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/qa/c_qa_tran_{pair_L1}_{pair_L2}.pkl"
            save_as_pickle(save_path, c_hidden_states)
    
    del model
    torch.cuda.empty_cache()