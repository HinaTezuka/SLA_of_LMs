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

def get_all_outputs_llama3_mistral(model, prompt, device):
    num_layers = model.config.num_hidden_layers
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(num_layers)]
    with TraceDict(model, MLP_act) as ret:
        with torch.no_grad():
            outputs = model(**prompt, output_hidden_states=True, output_attentions=True)
    MLP_act_values = [ret[act].output for act in MLP_act]
    
    return MLP_act_values

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
            act_values = get_all_outputs_llama3_mistral(model, inputs, device)
            for layer_idx in range(32):
                act_values_dict[layer_idx].append(act_values[layer_idx][:, -1, :].squeeze().detach().cpu().numpy()) # last layer only.
            
            c += 1
    
        activations_list = []
        for layer_idx in range(32):
            activations_list.append(np.mean(np.array(act_values_dict[layer_idx]), axis=0))

        # save as pkl.
        path_activations = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qa/{model_type}/act_values_{lang}'
        save_np_arrays(path_train, np.array(activations_list))

        del act_values_dict