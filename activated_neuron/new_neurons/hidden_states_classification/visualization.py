import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# LLaMA-3 モデル名とパス設定
llama3_model_names = {
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1",  # ja
    "nl": "ReBatch/Llama-3-8B-dutch",  # du
    "it": "DeepMount00/Llama-3-8b-Ita",  # ita
    "ko": "beomi/Llama-3-KoEn-8B",  # ko
}

# GPT-2 モデル名とパス設定
gpt2_model_names = {
    "ja": "rinna/japanese-gpt2-small", # ja
    "nl": "GroNLP/gpt2-small-dutch", # du
    "it": "GroNLP/gpt2-small-italian", # ita
    "ko": "skt/kogpt2-base-v2", # ko
}

# スコア格納用リスト
llama3_accuracies = []
gpt2_accuracies = []

# LLaMA-3 モデルのスコアを解凍して accuracy を格納
for L2, model_name in llama3_model_names.items():
    regression_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/logistic_regression/en_{L2}.pkl"
    
    # Pickleファイルを解凍して結果を取得
    with open(regression_path, "rb") as file:
        regression_results = pickle.load(file)
    
    # 各層ごとの accuracy を取得 (平均)
    layer_accuracies = [np.mean(result['accuracy']) for result in regression_results]  
    llama3_accuracies.append(layer_accuracies)

# GPT-2 モデルのスコアを解凍して accuracy を格納
for L2, model_name in gpt2_model_names.items():
    regression_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/gpt2/pickles/logistic_regression/en_{L2}.pkl"
    
    # Pickleファイルを解凍して結果を取得
    with open(regression_path, "rb") as file:
        regression_results = pickle.load(file)
    
    # 各層ごとの accuracy を取得 (平均)
    layer_accuracies = [np.mean(result['accuracy']) for result in regression_results]  
    gpt2_accuracies.append(layer_accuracies)

# GPT-2 の欠損部分 (13～32層) を NaN に埋める
gpt2_accuracies_padded = [np.concatenate([accuracies, np.full(32 - len(accuracies), np.nan)]) for accuracies in gpt2_accuracies]

# LLaMA-3 のスコアを 32層に合わせる (欠損値を NaN にする)
llama3_accuracies_padded = [np.concatenate([accuracies, np.full(32 - len(accuracies), np.nan)]) for accuracies in llama3_accuracies]

# スコア行列を作成 (列: モデル、行: 層)
score_matrix = np.array([*llama3_accuracies_padded, *gpt2_accuracies_padded])

# モデル名を設定
models = list(llama3_model_names.keys()) + list(gpt2_model_names.keys())  # LLaMA-3 と GPT-2 の言語

# ヒートマップ描画
plt.figure(figsize=(20, 6))
sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap="Purples",
            xticklabels=range(1, 33), yticklabels=models, vmin=0.90)
plt.title("Accuracy (LLaMA3 & GPT-2)")
plt.ylabel("Models")
plt.xlabel("Layer Index")
plt.savefig(
    f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/hidden_state_classification/test_acc.png',
    bbox_inches='tight',
    )