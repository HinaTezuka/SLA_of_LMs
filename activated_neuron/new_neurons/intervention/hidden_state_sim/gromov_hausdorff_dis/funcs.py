import sys
import dill as pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_hidden_states(model, tokenizer, device, data) -> list:
    """
    extract hidden states only for last tokens.
    is_norm: whether normalization of hidden states is required or not.

    return: np.array for input of sklearn classification models.
    """
    num_layers = 32
    # return
    output = {} # { sentence_idx: (L1_hidden_states_list, L2_hidden_states_list)}
    sentence_idx = 0

    for L1_txt, L2_txt in data:
        inputs_L1 = tokenizer(L1_txt, return_tensors="pt").to(device)
        inputs_L2 = tokenizer(L2_txt, return_tensors="pt").to(device)

        # get hidden_states
        with torch.no_grad():
            output_L1 = model(**inputs_L1, output_hidden_states=True)
            output_L2 = model(**inputs_L2, output_hidden_states=True)

        all_hidden_states_L1 = output_L1.hidden_states[1:]
        all_hidden_states_L2 = output_L2.hidden_states[1:]

        last_token_index_L1 = inputs_L1["input_ids"].shape[1] - 1
        last_token_index_L2 = inputs_L2["input_ids"].shape[1] - 1

        last_token_hidden_states_L1 = [
            layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy() for layer_hidden_state in all_hidden_states_L1[1:]
            ]
        last_token_hidden_states_L2 = [
            layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy() for layer_hidden_state in all_hidden_states_L2[1:]
            ]

        # hidden_states_L1 = [
        #     layer_hidden_state[0].detach().cpu().numpy() for layer_hidden_state in all_hidden_states_L1[1:]
        #     ]
        # hidden_states_L2 = [
        #     layer_hidden_state[0].detach().cpu().numpy() for layer_hidden_state in all_hidden_states_L2[1:]
        #     ]
        
        output[sentence_idx] = (last_token_hidden_states_L1, last_token_hidden_states_L2)
        sentence_idx += 1

    return output

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
