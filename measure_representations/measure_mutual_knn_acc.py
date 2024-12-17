import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

import mutual_knn_acc_funcs

""" Load models and tokenizers """
model_base_name = "meta-llama/Meta-Llama-3-8B"  # English(L1) model
# model_L2_name = "tokyotech-llm/Llama-3-Swallow-8B-v0.1"  # English + L2 model (Biltilingual)
model_L2_name = "DeepMount00/Llama-3-8b-Ita" # italian
# model_L2_name = "LiteLLMs/French-Alpaca-Llama3-8B-Instruct-v1.0-GGUF" # french
# model_L2_name = "DiscoResearch/Llama3-German-8B" # german
# model_L2_name = "beomi/Llama-3-KoEn-8B" # korean
# model_L2_name = "ReBatch/Llama-3-8B-dutch" # dutch

model_base = AutoModel.from_pretrained(model_base_name, output_hidden_states=True)
tokenizer_base = AutoTokenizer.from_pretrained(model_base_name)

model_L2 = AutoModel.from_pretrained(model_L2_name, output_hidden_states=True)
tokenizer_L2 = AutoTokenizer.from_pretrained(model_L2_name)

# [PAD] tokenを追加
if tokenizer_base.pad_token is None:
    tokenizer_base.add_special_tokens({'pad_token': '[PAD]'})
    model_base.resize_token_embeddings(len(tokenizer_base))  # モデルの語彙数を新しいトークンに合わせて調整([PAD]トークンを追加)←ノイズになりうる...?
if tokenizer_L2.pad_token is None:
    tokenizer_L2.add_special_tokens({'pad_token': '[PAD]'})
    model_L2.resize_token_embeddings(len(tokenizer_L2))  # モデルの語彙数を新しいトークンに合わせて調整


L2_iso_code = 'it'
# tatoeba datasetsから対訳テキストを取得
texts_en, texts_L2 = mutual_knn_acc_funcs.get_texts_from_translation_corpus(100, L2_iso_code)
# texts_en, texts_L2 = mutual_knn_acc_funcs.get_texts_from_translation_corpus(100, "gem", "en_ger")
# culclate mutual_knn_acc
mutual_knn_acc = mutual_knn_acc_funcs.compute_mutual_knn_acc(model_base, model_L2, tokenizer_base, tokenizer_L2, texts_en, texts_L2, 20)
print(f"Mutual KNN Accuracy: {mutual_knn_acc}")

# Load Tatoeba dataset
# dataset = load_dataset("tatoeba", lang1="en", lang2="du", split="train")  # Replace "L2" with the actual second language
# # dataset = load_dataset("KarthikSaran/trans_en_ger", split="train")
# print(dataset[0])
# # Select a subset of examples
# n_samples = 100  # Adjust this for the number of sentences
# # n_samples = 50 # for en-german corpus

# # texts_en = [sample['translation']['en'] for sample in dataset[:n_samples]]
# # texts_L2 = [sample['translation']['ja'] for sample in dataset[:n_samples]]  # Replace 'L2' with the actual second language key
# texts_en = [sample['translation']['en'] for sample in dataset.select(range(n_samples))]
# texts_L2 = [sample['translation']['du'] for sample in dataset.select(range(n_samples))]
# # Get representations from both models
# feats_base = mutual_knn_acc_funcs.extract_representations(model_base, tokenizer_base, texts_en)
# feats_L2 = mutual_knn_acc_funcs.extract_representations(model_L2, tokenizer_L2, texts_L2)

# # Calculate mutual KNN accuracy
# topk = 20  # Number of nearest neighbors
# mutual_knn_accuracy = mutual_knn_acc_funcs.mutual_knn(feats_base, feats_L2, topk)

# print(f"Mutual KNN Accuracy: {mutual_knn_accuracy}")

# with open("knn_acc_ko.txt", "w") as f:
#     f.write(f"Mutual KNN Accuracy: {mutual_knn_acc}\n")
