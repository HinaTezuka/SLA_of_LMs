import sys

import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

# モデルのロード（Llama 3）
model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device "cuda" if torch.cuda.is_available() else "cpu"

# 調整する100個の (layer_idx, neuron_idx) をランダムに選択
num_layers = len(model.model.layers)
num_vectors = 100  # 更新するベクトルの数
target_indices = []

# test (random)
# for _ in range(num_vectors):
#     layer_idx = random.randint(0, num_layers - 1)  # 層をランダムに選択
#     neuron_idx = random.randint(0, model.config.hidden_size - 1)  # ニューロン（行 or 列）をランダムに選択
#     target_indices.append((layer_idx, neuron_idx))

# 選択したベクトルだけをチューニング対象にする
for param in model.parameters():
    param.requires_grad = False  # すべてのパラメータを固定

vector_dict = {}  # 訓練したベクトルを保存する辞書
trainable_params = []  # optimizer に渡すパラメータリスト

for layer_idx, neuron_idx in target_indices:
    down_proj = model.model.layers[layer_idx].mlp.down_proj.weight  # (output_dim, hidden_dim)

    # ここでは行（出力次元）を対象にする
    target_vector = down_proj[neuron_idx]
    target_vector.requires_grad = True  # ここだけ学習可能にする

    vector_dict[(layer_idx, neuron_idx)] = target_vector  # 辞書に格納
    trainable_params.append(target_vector)  # optimizer に登録

# 翻訳タスク用データ
tokenizer = AutoTokenizer.from_pretrained(model_name)
source_texts = ["Hello, how are you?", "This is a test sentence."]
target_texts = ["こんにちは、お元気ですか？", "これはテストの文です。"]

inputs = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True).to(device)
labels = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

# 学習のセットアップ
optimizer = optim.AdamW(trainable_params, lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# チューニングループ
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss  # 翻訳タスクの損失

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # 学習後のベクトルを辞書に反映
    with torch.no_grad():
        for layer_idx, neuron_idx in target_indices:
            vector_dict[(layer_idx, neuron_idx)] = model.model.layers[layer_idx].mlp.down_proj.weight[neuron_idx].clone()

# 学習済みベクトルを保存
torch.save(vector_dict, "fine_tuned_vectors.pt")
print("チューニング済みの 100 ベクトルを保存しました！")