import os
import sys
sys.path.append('activated_neuron/new_neurons/transfer_neurons/txt_generation')
import pickle
import collections
import random
import json

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

from generation_funcs import (
    polywrite,
)

# load models (LLaMA3-8B).
model_names = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.3', 'CohereForAI/aya-expanse-8b']
model_name_dict = {
    'meta-llama/Meta-Llama-3-8B': 'Llama3-8B', 
    'mistralai/Mistral-7B-v0.3': 'Mistral-7B',
    'CohereForAI/aya-expanse-8b': 'Aya-expanse-8B',
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
langs = ['ja', 'nl', 'ko', 'it']
langs_for_polywrite = {
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
    'it': 'ita_Latn',
    'nl': 'nld_Latn',
}

for model_name in model_names:
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya'
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("MaLA-LM/PolyWrite", split="train")

    for lang in langs:
        data = ds.filter(lambda x: x["lang_script"]==langs_for_polywrite[lang])
        results = polywrite(model, tokenizer, device, data, lang)
    
        model_name_for_saving = model_name_dict[model_name]
        path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/txt_generation/results/{model_name_for_saving}_{lang}.json'
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


    del model
    torch.cuda.empty_cache()