import os
import sys

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

target_layers = [ _ for _ in range(1, 33)]
hidden_states = {}


""" model configs """
# LLaMA-3
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B"
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_names["ja"]).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_names["ja"])

# Set the prompt
prompt = "日本の首都はどこですか？"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Specify the layers
target_layer = 25  # Example: Use hidden_state from layer 10
last_layer = len(model.model.layers) - 1  # Index of the last layer

# Dictionary to store hidden states
stored_hidden_state = {}

# Hook function for the target layer
def hook_fn_target(module, input, output):
    """ Save hidden_state from the target layer """
    stored_hidden_state["ht"] = output
    return output  # Pass through the original output

# Hook function for the last layer
def hook_fn_last(module, input, output):
    """ Replace the last layer's hidden_state with the target layer's hidden_state """
    if "ht" in stored_hidden_state:
        return stored_hidden_state["ht"]  # Replace output with stored hidden_state
    return output  # Default output if the target hidden_state is not stored

# Register hooks
hook_target = model.model.layers[target_layer].register_forward_hook(hook_fn_target)
hook_last = model.model.layers[last_layer].register_forward_hook(hook_fn_last)

# Run inference

# generate()
# generated_outputs = model.generate(
#     inputs.input_ids, 
#     # max_new_tokens=50,
#     output_hidden_states=True,
#     return_dict_in_generate=True
# )

# inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # Extract logits

# Get the most probable token(s)
answer_ids = torch.argmax(logits, dim=-1)

# Decode the answer
answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
print("Answer:", answer)

# Remove hooks
hook_target.remove()
hook_last.remove()

# # Decode the generated text
# generated_text = tokenizer.decode(generated_outputs.sequences[0], skip_special_tokens=True)
# print("Generated Text:", generated_text)
