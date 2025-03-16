import os
import re
import random
import sys
import collections
import pickle

import torch
import numpy as np
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict

def compute_f1(a_gold, a_pred):
    # gold_toks = get_tokens(a_gold)
    # pred_toks = get_tokens(a_pred)
    gold_toks = a_gold
    pred_toks = a_pred
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

""" func for editing activation values """
def edit_activation(output, layer, layer_idx_and_neuron_idx):
    """
    edit activation value of neurons(indexed layer_idx and neuron_idx)
    output: activation values
    layer: sth like 'model.layers.{layer_idx}.mlp.act_fn'
    layer_idx_and_neuron_idx: list of tuples like [(layer_idx, neuron_idx), ....]
    """
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if str(layer_idx) in layer:  # layer名にlayer_idxが含まれているか確認
            output[:, :, neuron_idx] *= 0  # 指定されたニューロンの活性化値をゼロに設定

    return output

def mkqa_with_edit_activation(model, tokenizer, device, qa, L2, qa_num, layer_neuron_list):
    trace_layers = [f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:

        return mkqa(model, tokenizer, device, qa, L2, qa_num)

def mkqa(model, tokenizer, device, qa, L2: str, qa_num: int):
    c = 0 # question counter.
    f1_scores = []
    for i in range(len(qa['queries'])):
        if c == qa_num: break
        q = qa['queries'][i][L2] # question
        a = qa['answers'][i][L2][0]['text'] # answer
        if q == '' or q == None or  a == '' or a == None:
            continue

        # make prompt.
        if L2 == 'ja': prompt = f'{q} 答え: '
        elif L2 == 'nl': prompt = f'{q} Antwoord: '
        elif L2 == 'ko': prompt = f'{q} 답변: '
        elif L2 == 'it': prompt = f'{q} Risposta: '

        # run inference.
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        pre = tokenizer.decode(output[0], skip_special_tokens=True)
        # 
        if L2 == 'ja': pre = pre.split("答え: ")[-1].strip()
        if L2 == 'nl': pre = pre.split('Antwoord: ')[-1].strip()
        if L2 == 'ko': pre = pre.split('답변: ')[-1].strip()
        if L2 == 'it': pre = pre.split('Risposta: ')[-1].strip()
        f1 = compute_f1(a, pre)
        f1_scores.append(f1)
        c += 1
        print(f'question: {q}')
        print(f'model ans: {pre}')
        print(f'gorund truth: {a}')
        print(f'f1: {f1}')
    
    return np.mean(np.array(f1_scores))

def save_as_pickle(file_path: str, target_dict) -> None:
    """
    Save a dictionary as a pickle file with improved safety.
    """
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temp_path = file_path + ".tmp"  # Temporary file for safe writing

    try:
        # Write to a temporary file
        with open(temp_path, "wb") as f:
            pickle.dump(target_dict, f)
        # Replace the original file with the temporary file
        os.replace(temp_path, file_path)
        print("pkl_file successfully saved.")
    except Exception as e:
        # Clean up temporary file if an error occurs
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e  # Re-raise the exception for further handling

def unfreeze_pickle(file_path: str):
    """
    Load a pickle file as a dictionary with error handling.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Error unpickling file {file_path}: {e}")