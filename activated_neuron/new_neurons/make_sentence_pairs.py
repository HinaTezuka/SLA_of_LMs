import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset
from transformers import AutoTokenizer

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

""" model configs """
# LLaMA-3
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}
device = "cuda" if torch.cuda.is_available() else "cpu"

for L2, model_name in model_names.items():
    L1 = "en" # L1 is fixed to english.

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    """ tatoeba translation corpus """
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    num_sentences = 2000
    dataset = dataset.select(range(num_sentences))
    tatoeba_data = []
    token_lens = []

    for item in dataset:
        en_tokens = tokenizer(item['translation'][L1], return_tensors="pt")
        token_lens.append(len(en_tokens['input_ids'][0]))
        # check if there are empty sentences.
        if item['translation'][L1] != '' and item['translation'][L2] != '':
            tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
    tatoeba_data_len = len(tatoeba_data)
    en_ave_tokens = round(np.array(token_lens).mean())

    """
    baseとして、対訳関係のない1文ずつのペアを作成
    (L1(en)はhttps://huggingface.co/datasets/agentlans/high-quality-english-sentences,
    L2はtatoebaの該当データを使用)
    """
    random_data = []
    # L1(en)
    en_base_ds = load_dataset("Devavrat28/English-Marathi_Complex_Sentences")
    random_data_en = en_base_ds["train"]
    en_base_ds_idx = 0
    for item in dataset:
        while True:
            en_text = random_data_en["English"][en_base_ds_idx]
            en_tokens = tokenizer(en_text, return_tensors="pt")
            token_len = len(en_tokens['input_ids'][0])
            print(token_len)
            if en_text != '' and item['translation'][L2] != '' and token_len <= en_ave_tokens:
                random_data.append((en_text, item["translation"][L2]))
                print(en_text)
                en_base_ds_idx += 1
                break
            en_base_ds_idx += 1

    """ save as pickle file. """
    same_semantics_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/sentence_pairs/same_semantics/{L2}.pkl'
    diff_semantics_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/sentence_pairs/different_semantics/{L2}.pkl'
    save_as_pickle(same_semantics_path, tatoeba_data)
    save_as_pickle(diff_semantics_path, random_data)

    """ unfreeze """
    print(f'============== {L2} ===============')
    same = unfreeze_pickle(same_semantics_path)
    print(f'len_same_semantics: {len(same)}')
    print(same[:100])
    diff = unfreeze_pickle(diff_semantics_path)
    print(f'len_diff_semantics: {len(diff)}')
    print(diff[-100:])


