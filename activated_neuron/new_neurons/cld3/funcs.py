import os
import sys
import dill as pickle
import json

import numpy as np
import torch
import torch.nn.functional as F
from baukit import TraceDict

def project_hidden_emb_to_vocab(model, tokenizer, hidden_states: torch.Tensor, last_token_index: int, top_k: int) -> dict:
    token_preds_dict = {} # {layer_idx: [token1, token2, ...]}
    for layer_idx in range(1, 33): # <- remove embedding layer(0th layer).
        hidden_state = hidden_states[layer_idx][0, last_token_index]
        # normalization
        normed = model.model.norm(hidden_state)
        # get logits
        logits = torch.matmul(model.lm_head.weight, normed.T)
        # make distribution
        probs = F.softmax(logits, dim=-1)
        probs = torch.reshape(probs, (-1,)).cpu().detach().numpy()

        probs_ = [] # [ (token_idx, probability), (), ..., () ]
        for index, prob in enumerate(probs):
            probs_.append((index, prob))

        top_k_tokens = sorted(probs_, key=lambda x: x[1], reverse=True)[:top_k]
        token_preds = [tokenizer.decode(t[0]) for t in top_k_tokens]

        token_preds_dict[layer_idx] = token_preds

    return token_preds_dict

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

def get_hidden_states_with_edit_activation(model, inputs, layer_neuron_list):
    trace_layers = [f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:
        # run inference
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
    return output.hidden_states

def print_tokens(token_dict):
    for k, v in token_dict.items():
        print(f"=========================== {k} ===========================")
        print(v, "\n")

def save_as_pickle(file_path, target_dict) -> None:
    """
    save dict as pickle file.
    """
    # directoryを作成（存在しない場合のみ)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(target_dict, f)

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

def save_as_json(data: dict, file_name_with_path: str) -> None:
    # Check if the directory exists (create it if not)
    output_dir = os.path.dirname(file_name_with_path)
    os.makedirs(output_dir, exist_ok=True)

    temp_file_path = file_name_with_path + ".tmp"

    try:
        # Convert keys to strings for serialization
        serializable_data = {str(key): value for key, value in data.items()}
        # Write data to the temporary file
        with open(temp_file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=4)

        # If writing is successful, rename the temporary file to the original file
        os.rename(temp_file_path, file_name_with_path)
        print("Saving completed.")

    except Exception as e:
        # Error handling: remove the temporary file if it exists
        print(f"Error saving JSON: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def unfreeze_json(file_name_with_path: str) -> dict:
    if not os.path.exists(file_name_with_path):
        raise FileNotFoundError(f"JSON file not found: {file_name_with_path}")

    try:
        with open(file_name_with_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Convert string keys back to tuples
        return {eval(key): value for key, value in data.items()}
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON file {file_name_with_path}: {e}")


if __name__ == "__main__":
    print()