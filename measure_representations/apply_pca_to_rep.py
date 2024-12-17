import sys
sys.path.append('/home/s2410121/proj_LA/measure_representations')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

# from measure_mutual_knn_acc import *
from mutual_knn_acc_funcs import *

def extract_representations(model, tokenizer, texts, layer_idx=-1) -> torch.Tensor: # layer_indexで「どの層」のrepresentationsをとるかを指定
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
        hidden_states = outputs.hidden_states  # List of layer-wise hidden states
        # print(hidden_states)
        representations = hidden_states[layer_idx]  # Extract the chosen layer
        # print(representations)
    # return representations.mean(dim=1)  # Mean-pooling across tokens
    return F.normalize(representations.mean(dim=1))

""" PCA """

# iso_code
L2_iso_code = {
    'en': 'en',
    'ja': 'ja',
    'ita': 'it',
    'du': 'nl',
}

""" LLAMA3 """
""" base model(en) """
llama3_en_name = "tokyotech-llm/Llama-3-Swallow-8B-v0.1"
llama3_en = AutoModel.from_pretrained(llama3_en_name, output_hidden_states=True)
tokenizer_llama3_en = AutoTokenizer.from_pretrained(llama3_en_name)
""" japanese """
llama3_ja_name = "tokyotech-llm/Llama-3-Swallow-8B-v0.1"
llama3_ja = AutoModel.from_pretrained(llama3_ja_name, output_hidden_states=True)
tokenizer_llama3_ja = AutoTokenizer.from_pretrained(llama3_ja_name)

# [PAD] tokenを追加
if tokenizer_llama3_en.pad_token is None:
    tokenizer_llama3_en.add_special_tokens({'pad_token': '[PAD]'})
    llama3_en.resize_token_embeddings(len(tokenizer_llama3_en))  # モデルの語彙数を新しいトークンに合わせて調整([PAD]トークンを追加)←ノイズになりうる...?
if tokenizer_llama3_ja.pad_token is None:
    tokenizer_llama3_ja.add_special_tokens({'pad_token': '[PAD]'})
    llama3_ja.resize_token_embeddings(len(tokenizer_llama3_ja))  # モデルの語彙数を新しいトークンに合わせて調整

""" 特定の層のrepresentationsを取得 """
layer_idx = -1

# それぞれのrepresentationsを取得
en_txt, ja_txt = get_texts_from_translation_corpus(1, L2_iso_code['ja'])
rep_en = extract_representations(llama3_en, tokenizer_llama3_en, en_txt, layer_idx)
print(rep_en)
rep_ja = extract_representations(llama3_ja, tokenizer_llama3_ja, ja_txt, layer_idx)

# 取得したrepresentationsをnp.arrayに変換・concatenate
np_arr_rep_en = np.array(rep_en)
print(np_arr_rep_en)
np_arr_rep_ja = np.array(rep_ja)
rep_en_ja = np.concatenate((np_arr_rep_en, np_arr_rep_ja), axis=0)

# print(rep_en_ja, np_arr_rep_en, np_arr_rep_ja)
# 標準化
scaler = StandardScaler()
# representationの平均と標準偏差を計算して標準化
rep_en_ja = scaler.fit_transform(rep_en_ja)

pca = PCA(n_components=2)
pca_rep_en_ja = pca.fit_transform(rep_en_ja)
# pca_lb = pca.transform(emb_lb)

# ラベルの作成
labels = ['en'] * np_arr_rep_en.shape[0] + ['ja'] * np_arr_rep_ja.shape[0]

# DataFrameに変換してラベルを追加
df_pca = pd.DataFrame(data=pca_rep_en_ja, columns=['PC1', 'PC2'])
df_pca['labels'] = labels

# 可視化
fig, ax = plt.subplots(figsize=(8, 8))
targets = ['en', 'ja']
colors = ['r', 'b']

for target, color in zip(targets, colors):
    indicesToKeep = df_pca['labels'] == target
    ax.scatter(df_pca.loc[indicesToKeep, 'PC1'], df_pca.loc[indicesToKeep, 'PC2'], c=color, s=50, label=target)

ax.set_xlabel('PC1', fontsize=15)
ax.set_ylabel('PC2', fontsize=15)
ax.set_title('PCA of English and Japanese Representations', fontsize=20)
ax.legend(targets)
ax.grid()

# プロットの保存
fig.savefig('/home/s2410121/proj_LA/measure_representations/pca_plot.png')
plt.show()
