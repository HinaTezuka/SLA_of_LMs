import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from datasets import load_dataset, get_dataset_config_names
import torch
import pandas as pd

# 使用するモデル名のリスト
model_names = [
                # llama3-8b
                # "meta-llama/Meta-Llama-3-8B", # en
                # "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
                # "DiscoResearch/Llama3-German-8B", # ger
                # "DeepMount00/Llama-3-8b-Ita", # ita
                # "beomi/Llama-3-KoEn-8B", # ko
                "ReBatch/Llama-3-8B-dutch", # du
              ]

# BLiMPの評価項目リスト
configs = get_dataset_config_names("blimp")

# 評価関数
# def evaluate_sentence_pair(model, tokenizer, sentence1, sentence2):
#     inputs1 = tokenizer(sentence1, return_tensors="pt")
#     inputs2 = tokenizer(sentence2, return_tensors="pt")

#     with torch.no_grad():
#         outputs1 = model(**inputs1)
#         outputs2 = model(**inputs2)

#     """ モデルがそれぞれの文を生成する確率 """
#     # score1 = outputs1.logits.log_softmax(dim=-1)[..., inputs1.input_ids[0]].sum()
#     # score2 = outputs2.logits.log_softmax(dim=-1)[..., inputs2.input_ids[0]].sum()

#     # 文1の対数確率をトークンごとに取得して平均
#     log_probs1 = outputs1.logits.log_softmax(dim=-1)
#     score1 = log_probs1[..., inputs1.input_ids[0]].mean()

#     # 文2の対数確率をトークンごとに取得して平均
#     log_probs2 = outputs2.logits.log_softmax(dim=-1)
#     score2 = log_probs2[..., inputs2.input_ids[0]].mean()

#     return score1, score2

def evaluate_sentence_pair(model, tokenizer, sentence1, sentence2):
    inputs1 = tokenizer(sentence1, return_tensors="pt").to("cuda")
    inputs2 = tokenizer(sentence2, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # 文1の対数確率をトークンごとに取得して平均
    log_probs1 = outputs1.logits.log_softmax(dim=-1)
    score1 = 0.0
    for i in range(inputs1.input_ids.size(1) - 1):
        target_token_id = inputs1.input_ids[0, i + 1]
        score1 += log_probs1[0, i, target_token_id].item()
    score1 /= (inputs1.input_ids.size(1) - 1)  # 平均を取る

    # 文2の対数確率をトークンごとに取得して平均
    log_probs2 = outputs2.logits.log_softmax(dim=-1)
    score2 = 0.0
    for i in range(inputs2.input_ids.size(1) - 1):
        target_token_id = inputs2.input_ids[0, i + 1]
        score2 += log_probs2[0, i, target_token_id].item()
    score2 /= (inputs2.input_ids.size(1) - 1)  # 平均を取る

    return score1, score2

# データを保存するリスト
results = []

# 各モデルについてBLiMPのタスクを評価
for model_name in model_names:
    # モデルとトークナイザーをロード
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 各評価項目ごとに評価
    for config in configs:
        blimp = load_dataset("blimp", config)
        correct = 0
        total = 0

        for example in blimp["train"]:
            sentence1 = example["sentence_good"]
            sentence2 = example["sentence_bad"]
            score1, score2 = evaluate_sentence_pair(model, tokenizer, sentence1, sentence2)

            if score1 > score2:
                correct += 1
            total += 1

        # 精度を計算して結果を保存
        accuracy = correct / total
        results.append({
            "Model": model_name,
            "Task": config,
            "Accuracy": accuracy
        })

# データフレームに変換
df = pd.DataFrame(results)
print(df)

# print(df)
# CSVに保存
df.to_csv("/home/s2410121/proj_LA/gpt2-small_blimp/csv_files_final/llam3_en_du.csv", index=False)

print("評価結果をcsv fileに保存しました。")
