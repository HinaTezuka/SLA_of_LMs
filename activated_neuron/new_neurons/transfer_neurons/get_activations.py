"""
detect language specific neurons
"""
import sys
import dill as pickle

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    multilingual_dataset,
    track_neurons_with_text_data,
    save_as_pickle,
)

# making multilingual data.
langs = ["ja", "nl", "ko", "it", "en"]
multilingual_sentences = multilingual_dataset(langs) # 2500 sentences(500 for each lang).

# LLaMA-3(8B) models.
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}
device = "cuda" if torch.cuda.is_available() else "cpu"

""" get activaitons and save as pkl. """
for L2, model_name in model_names.items():
    # model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    activations = track_neurons_with_text_data(model, device, tokenizer, multilingual_sentences)

    # save activations as pickle file.
    save_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/activations/{L2}.pkl"
    save_as_pickle(save_path, activations)
    print(f"successfully saved activations of {L2} model as pkl.")

    # clean cache.
    del model
    torch.cuda.empty_cache()