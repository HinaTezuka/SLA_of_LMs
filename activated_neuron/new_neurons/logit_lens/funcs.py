import os
import sys
import dill as pickle
import json

import numpy as np
import torch
import torch.nn.functional as F


def project_value_to_vocab(model, tokenizer, layer_idx, value_idx, top_k=10, normed=False):
  value_vector = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[value_idx]
  # normalization if it's required.
  value_vector = model.model.norm(value_vector) if normed else value_vector
  # get logits.
  logits = torch.matmul(model.lm_head.weight, value_vector.T)
  # make distribution based on logits.
  probs = F.softmax(logits, dim=-1)
  probs = torch.reshape(probs, (-1,)).cpu().detach().numpy()

  probs_ = [] # [ (token_idx, probability), (), ..., () ]
  for index, prob in enumerate(probs):
      probs_.append((index, prob))

  top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:top_k]
  value_preds = [(tokenizer.decode(t[0]), t[0]) for t in top_k]

  return value_preds

def get_embedding_for_token(token_idx):
    # get embeddings from token_idx.
    embedding = model.model.embed_tokens.weight[token_idx]
    return embedding

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