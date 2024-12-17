import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from datasets import load_dataset, get_dataset_config_names
import torch
import pandas as pd

model_name = "rinna/japanese-gpt2-small"
jblimp = load_dataset("polm-stability/jblimp")
device = "cuda" if torch.cuda.is_available() else "cpu"
# for ex in jblimp["train"]:
#     print(ex)
# sys.exit()

def evaluate_sentence_pair(model, tokenizer, sentence1, sentence2):
    inputs1 = tokenizer(sentence1, return_tensors="pt").to(device)
    inputs2 = tokenizer(sentence2, return_tensors="pt").to(device)

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
predictions = defaultdict(lambda: defaultdict(int))
"""
predictions:
{
"task_name":
        "total": num_of_total_sentence(int)
        "correct": num_of_correct_predictions(int)
}
"""
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

for example in jblimp["train"]:
    task_name = example["phenomenon"]
    sentence1 = example["good_sentence"]
    sentence2 = example["bad_sentence"]
    score1, score2 = evaluate_sentence_pair(model, tokenizer, sentence1, sentence2)
    if score1 > score2:
        predictions[task_name]["correct"] += 1
    predictions[task_name]["total"] += 1

# 精度を計算して結果を保存
for task_name, scores in predictions.items():
    results.append({
        "Model": model_name,
        "Task": task_name,
        "Accuracy": scores["correct"] / scores["total"]
    })

df = pd.DataFrame(results)
print(df)

# CSVに保存
df.to_csv("/home/s2410121/proj_LA/gpt2-small_blimp/jblimp/csv_files/gpt2_jblimp.csv", index=False)

print("評価結果をcsv fileに保存しました。")

