import os
import sys
import random

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from funcs import (
    save_as_pickle,
    unfreeze_pickle,
    save_np_arrays,
    unfreeze_np_arrays,
)

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- model and tokenizer ---
model_name = "meta-llama/Meta-Llama-3-8B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
score_type = 'cos_sim'

# --- 対象ベクトルの指定 ---
model_type = "llama3"
path_type_1 = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/{score_type}/ja_mono_train.pkl"
path_type_2 = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/ja_sorted_neurons.pkl"
type_1 = unfreeze_pickle(path_type_1)[:1000]
type_2 = unfreeze_pickle(path_type_2)
type_2 = [neuron for neuron in type_2 if neuron[0] in range(20, 32)][:1000]
target_indices = type_1 + type_2

# --- proxy vector の準備 ---
vector_dict = {}
trainable_params = []

for layer_idx, neuron_idx in target_indices:
    original_vector = model.model.layers[layer_idx].mlp.down_proj.weight[:, neuron_idx].detach().clone()
    proxy_param = torch.nn.Parameter(original_vector)
    vector_dict[(layer_idx, neuron_idx)] = proxy_param
    trainable_params.append(proxy_param)

# --- forward hook 登録 ---
hook_handles = []

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        weight = module.weight.clone()
        for (l_idx, n_idx), vec in vector_dict.items():
            if l_idx == layer_idx:
                weight[:, n_idx] = vec
        return torch.nn.functional.linear(input[0], weight, module.bias)
    return hook_fn

for (layer_idx, _) in set(target_indices):
    handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(make_hook(layer_idx))
    hook_handles.append(handle)

# --- データセット定義 ---
class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return input_ids, attention_mask

path_mono_train = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/ja_mono_train.pkl"
sentences = unfreeze_pickle(path_mono_train)
dataset = SentenceDataset(sentences, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# --- optimizer と scheduler ---
optimizer = optim.AdamW(trainable_params, lr=2e-5)
num_epochs = 5
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(dataloader))
# scaler = GradScaler()

# --- チューニングループ ---
model.train()

for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    epoch_loss = 0.0

    for input_ids, attention_mask in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = input_ids.clone()

        optimizer.zero_grad()

        # with autocast():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # scheduler.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
    save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/tuned_weights/fine_tuned_vectors_epoch{epoch+1}.pt'
    torch.save(vector_dict, save_path)
    print(f"Saved weights for epoch {epoch+1}")

    torch.cuda.empty_cache()

# --- フック削除 ---
for handle in hook_handles:
    handle.remove()

print("Training completed and vectors saved.")