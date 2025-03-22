import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict

from qa_funcs import (
    save_np_arrays,
    unfreeze_pickle,
    unfreeze_np_arrays,
)
from funcs import (
    # get_hidden_states,
)

c = 0 # question counter.
ans_patterns = {
'ja': '答え: ',
'nl': 'Antwoord: ',
'ko': '답변: ',
'it': 'Risposta: ',
}
models = {
    'llama3': 'meta-llama/Meta-Llama-3-8B',
    'mistral': 'mistralai/Mistral-7B-v0.3',
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
langs = ['ja', 'nl', 'ko', 'it']

# load QA dataset.
qa = load_dataset('apple/mkqa')['train']
qa = qa.shuffle(seed=42)

def get_hidden_states(model, tokenizer, device, num_layers, data):
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

    return dict(c_hidden_states)

def get_centroid_of_shared_space(hidden_states: dict):
    centroids = [] # [c1, c2, ] len = layer_num(32layers: 0-31)
    for layer_idx, c in hidden_states.items():
        final_c = np.mean(c, axis=0) # calc mean of c(shared point per text) of all text.
        centroids.append(final_c)
    return centroids

qa_num = 1000 # <- 2000までは　train　set.
c = 1
for model_type, model_name in models.items():
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for lang in langs:
        act_values_dict = defaultdict(list) # {layer_idx: [act_values_sample1, act_values_sample2, ...]}
        for i in range(len(qa['queries'])):
            if c == qa_num+1: break
            q = qa['queries'][i][lang] # question
            a = qa['answers'][i][lang][0]['text'] # ans

            if any(not v for v in [q, a]):
                continue

            # make prompt.
            prompt = f'{q}? {ans_patterns[lang]}'

            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            # get act_values for lang_act.
            act_values = get_hidden_states(model, inputs, device)
            for layer_idx in range(32):
                act_values_dict[layer_idx].append(act_values[layer_idx][:, -1, :].squeeze().detach().cpu().numpy()) # last layer only.
            
            c += 1
    
        activations_list = []
        for layer_idx in range(32):
            activations_list.append(np.mean(np.array(act_values_dict[layer_idx]), axis=0))

        # save as pkl.
        path_activations = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qa/{model_type}/act_values_{lang}'
        save_np_arrays(path_activations, np.array(activations_list))

        del act_values_dict