import os
import re
import random
import string
import sys
import collections
import pickle
import math
from collections import Counter, defaultdict
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict


def make_prompt(prompt: str, L2: str) -> str:
    question_patterns = {
        'ja': '質問: ',
        'nl': 'Vraag: ',
        'ko': '질문: ',
        'it': 'Domanda: ',
        'en': 'Question: ',
    }
    ans_patterns = {
    'ja': '答え: ',
    'nl': 'Antwoord: ',
    'ko': '답변: ',
    'it': 'Risposta: ',
    'en': 'Answer: ',
    }
    return_prompt = f"""
    {question_patterns[L2]}{prompt}
    {ans_patterns[L2]}
    """

    return return_prompt

""" func for editing activation values """
def edit_activation(output, layer, layer_idx_and_neuron_idx):
    """
    edit activation value of neurons(indexed layer_idx and neuron_idx)
    output: activation values
    layer: sth like 'model.layers.{layer_idx}.mlp.act_fn'
    layer_idx_and_neuron_idx: list of tuples like [(layer_idx, neuron_idx), ....]
    """
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        # if str(layer_idx) in layer and output.shape[1] != 1:
        if f"model.layers.{layer_idx}." in layer and output.shape[1] != 1:
            output[:, -1, neuron_idx] *= 0

    return output

def edit_activation_always(output, layer, layer_idx_and_neuron_idx):
    """
    edit activation value of neurons(indexed layer_idx and neuron_idx)
    output: activation values
    layer: sth like 'model.layers.{layer_idx}.mlp.act_fn'
    layer_idx_and_neuron_idx: list of tuples like [(layer_idx, neuron_idx), ....]
    """
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        # if str(layer_idx) in layer and output.shape[1] != 1:
        if f"model.layers.{layer_idx}." in layer:
            output[:, -1, neuron_idx] *= 0

    return output

def polywrite_with_edit_activation_always(model, tokenizer, device, data, L2, layer_neuron_list, num_samples=50):
    trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]))
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation_always(output, layer, layer_neuron_list)) as tr:

        return polywrite(model, tokenizer, device, data, L2, num_samples)

def polywrite_with_edit_activation(model, tokenizer, device, data, L2, layer_neuron_list, num_samples=50):
    trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]))
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:

        return polywrite(model, tokenizer, device, data, L2, num_samples)

def polywrite(model, tokenizer, device, data, L2, num_samples=50):
    results = []
    for sample_idx, sample in enumerate(data):
        if sample_idx == num_samples: break
        prompt = make_prompt(sample['prompt_translated'], L2)
        input = tokenizer(prompt, return_tensors='pt').to(device)
        input_len = input['input_ids'].shape[1] # nums of input tokens.

        with torch.no_grad():
            output = model.generate(
                    **input,
                    do_sample=False,
                    # max_new_tokens=512,
                )

        output_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
        results.append({
            'sample_idx': sample_idx,
            'input_lang': L2,
            'output': output_text,
        })
    
    return results

def save_np_arrays(save_path, np_array):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Save directly to .npz
        np.savez(save_path, data=np_array)
        print(f"Array successfully saved to {save_path}")
    except Exception as e:
        print(f"Failed to save array: {e}")

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

def unfreeze_np_arrays(save_path):
    try:
        with np.load(save_path) as data:
            return data["data"]
    except Exception as e:
        print(f"Failed to load array: {e}")
        return None


def sentence_average_entropy(generated_tokens, scores): # 各トークンの平均エントロピー.
    total_entropy = 0.0
    for i, score in enumerate(scores):
        probs = F.softmax(score, dim=-1)
        log_probs = F.log_softmax(score, dim=-1)
        token_id = generated_tokens[i].item()
        token_entropy = -log_probs[0, token_id].item()  # -log P(token_i)
        total_entropy += token_entropy

    avg_entropy = total_entropy / len(generated_tokens)
    return avg_entropy

def sentence_distribution_entropy(generated_tokens, scores):
    total_entropy = 0.0
    for i, score in enumerate(scores):
        probs = F.softmax(score, dim=-1)
        log_probs = F.log_softmax(score, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()  # 全トークンにまたがるエントロピー
        total_entropy += entropy

    avg_entropy = total_entropy / len(generated_tokens)
    return avg_entropy

def sentence_perplexity(generated_tokens, scores):
    total_entropy = 0.0
    for i, score in enumerate(scores):
        probs = F.softmax(score, dim=-1)
        log_probs = F.log_softmax(score, dim=-1)
        token_id = generated_tokens[i].item()
        token_entropy = -log_probs[0, token_id].item()  # -log P(token_i)
        total_entropy += token_entropy

    avg_entropy = total_entropy / len(generated_tokens)
    perplexity = math.exp(avg_entropy)
    return perplexity

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