"""
calc centroid.
centroid: center point of Language-Agnostic Space.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    monolingual_dataset_en,
    get_hidden_states_en_only,
    get_centroid_of_shared_space,
    save_as_pickle,
    unfreeze_pickle,
)

langs = ["ja", "nl", "ko", "it", "en"]
# LLaMA3-8B / Mistral-7B
model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", 'CohereForAI/aya-expanse-8b']
model_names = ["mistralai/Mistral-7B-v0.3", 'CohereForAI/aya-expanse-8b']
device = "cuda" if torch.cuda.is_available() else "cpu"
num_layers = 32

# load qa data and shuffle.
qa_num = 1000
qa = load_dataset('apple/mkqa')['train']
qa = qa.shuffle(seed=42)

# centroids of english texts.
centroids = {} # { L2: [shared_centroids(en-L2)_1, ...} <- len(values) = 32(layer_num).

def get_sentences_qa(qa, L2: str):
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

for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    for L2 in langs:
        # hidden_states for each L2.
        # mono L2 sentences.
        sentences = get_sentences_qa(qa, L2)
        # save qa sentences.
        save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/sentences.pkl'
        save_as_pickle(save_path, sentences)
        
        # get centroids of hidden_states(of en-L2 sentence pairs).
        c_hidden_states = get_hidden_states_en_only(model, tokenizer, device, num_layers, sentences)
        shared_space_centroids = get_centroid_of_shared_space(c_hidden_states) # list: [c_layer1, c_layer2, ...]

        # save centroids as pkl.
        save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_{L2}_qa.pkl"
        save_as_pickle(save_path, shared_space_centroids)
        print(f'{L2} completed.')
    
    del model
    torch.cuda.empty_cache()