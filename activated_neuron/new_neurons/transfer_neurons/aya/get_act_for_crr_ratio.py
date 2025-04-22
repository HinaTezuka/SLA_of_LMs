import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    multilingual_dataset_for_lang_specific_detection,
    save_as_pickle,
    unfreeze_pickle,
    save_np_arrays,
)

def track_neurons_with_text_data_elem_wise(model, device, tokenizer, data):
    # layers_num
    num_layers = 32
    # nums of total neurons (per a layer)
    num_neurons = 14336
    # returning np_array.
    activation_array = np.zeros((num_layers, num_neurons, len(data)))
    """ """
    for i, text in enumerate(data):
        # input text
        inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)
        token_len = len(inputs[0])

        """ get activations. """
        act_values = [] # len: layer_num
        # hook fn
        def get_elem_wise_product(model, input):
            act_values.append(input[0][0][-1].detach().cpu().numpy()) # last tokenに対応する活性値のみ取得
        handles = []
        for layer in model.model.layers:
            handle = layer.mlp.down_proj.register_forward_pre_hook(get_elem_wise_product)
            handles.append(handle)
        # run inference.
        with torch.no_grad():
            output = model(inputs)
        # remove hook
        for handle in handles:
            handle.remove()

        for layer_idx in range(num_layers):
            for neuron_idx in range(num_neurons):
                activation_array[layer_idx, neuron_idx, i] = act_values[layer_idx][neuron_idx] # i: question_idx

    return activation_array

# making multilingual data.
langs = ["ja", "nl", "ko", "it", "en"]
num_sentences = 1000
start_indics = {
    "ja": 0,
    "nl": 1000,
    "ko": 2000,
    "it": 3000,
    "en": 4000,
}
multilingual_sentences = multilingual_dataset_for_lang_specific_detection(langs, num_sentences) # 2500 sentences(500 for each lang).

# model and tokenizer.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'CohereForAI/aya-expanse-8b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model_langs = ["ja", "nl", "ko", "it", "en"]

""" get activaitons and save as npz and pkl. """
for L2 in model_langs:
    # start and end indices.
    start_idx = start_indics[L2]
    end_idx = start_idx + num_sentences
    # get activations and corresponding labels.
    activations = track_neurons_with_text_data_elem_wise(
        model, 
        device, 
        tokenizer, 
        multilingual_sentences[start_idx:end_idx], 
        )
    # save activations as pickle file.
    path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/aya/activations/{L2}_activations_crr_ratio'
    save_np_arrays(path, activations)
    print(f'done: {L2}')