from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from datasets import load_dataset, get_dataset_config_names
import torch
import pandas as pd

# 使用するモデル名のリスト
model_names = [
                "gpt2", # base model: original gpt2(small) model : en
                "rinna/japanese-gpt2-small", # ja
                # "GroNLP/gpt2-small-dutch", # du
                # "dbmdz/german-gpt2", # ger
                # "GroNLP/gpt2-small-italian", # ita
                # "dbddv01/gpt2-french-small", # fre
                # "skt/kogpt2-base-v2", # ko
                # "datificate/gpt2-small-spanish", # spa
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
    inputs1 = tokenizer(sentence1, return_tensors="pt")
    inputs2 = tokenizer(sentence2, return_tensors="pt")

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
    model = AutoModelForCausalLM.from_pretrained(model_name)
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

# 各モデルごとに正答率の平均を計算
overall_accuracy = df.groupby('Model')['Accuracy'].mean().reset_index()
print(overall_accuracy)

# 列名を変更してOVERALLにします
overall_accuracy.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)


# CSVに保存
df.to_csv("blimp_evaluation_results_complete2_gpt2_all_final_en_ja.csv", index=False)

print("評価結果をcsv fileに保存しました。")
