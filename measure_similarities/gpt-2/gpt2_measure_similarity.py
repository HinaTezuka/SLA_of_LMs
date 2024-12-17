"""
memo:
・GPT-2-smallを対象に調査

GPT2Model(
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)
  (drop): Dropout(p=0.1, inplace=False)
  (h): ModuleList(
    (0-11): 12 x GPT2Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attn): GPT2SdpaAttention(
        (c_attn): Conv1D()
        (c_proj): Conv1D()
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): GPT2MLP(
        (c_fc): Conv1D()
        (c_proj): Conv1D()
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
"""
import sys
sys.path.append('/home/s2410121/proj_LA/measure_similarities')

from similarity_funcs import *

import torch

from collections import defaultdict
from transformers import AutoModel, GPT2Model, AutoModelForCausalLM, GenerationConfig

""" モデル指定・ロード """

# モデルの指定
gpt2_model_original_name = "gpt2" # original gpt2(small) model : en
gpt2_model_japanese_name = "rinna/japanese-gpt2-small" # ja
gpt2_model_dutch_name = "GroNLP/gpt2-small-dutch" # du
gpt2_model_german_name = "dbmdz/german-gpt2" # ger
# gpt2_model_german_name = "ml6team/gpt2-small-german-finetune-oscar" # ger
gpt2_model_italian_name = "GroNLP/gpt2-small-italian" # ita
gpt2_model_french_name = "dbddv01/gpt2-french-small" # fre
gpt2_model_korean_name = "skt/kogpt2-base-v2" # ko
gpt2_model_spanish_name = "datificate/gpt2-small-spanish" # spa

# モデルのロード
gpt2_model_original = GPT2Model.from_pretrained(gpt2_model_original_name) # en
gpt2_model_ja = GPT2Model.from_pretrained(gpt2_model_japanese_name) # ja
gpt2_model_du = GPT2Model.from_pretrained(gpt2_model_dutch_name) # du
gpt2_model_ger = GPT2Model.from_pretrained(gpt2_model_german_name) # ger
gpt2_model_ita = GPT2Model.from_pretrained(gpt2_model_italian_name) # ita
gpt2_model_fre = GPT2Model.from_pretrained(gpt2_model_french_name) # fre
gpt2_model_ko = GPT2Model.from_pretrained(gpt2_model_korean_name) # ko
gpt2_model_spa = GPT2Model.from_pretrained(gpt2_model_spanish_name) # spa

""" それぞれのモデルのparametersを取得 """

# parametersを取得 <- いくつかの辞書ではtransformerという接頭辞がついているため、記法を合わせるために削除(similarity_funcs.pyのdelete_transformer_prefixes_from_state_dict_keysを使用)
state_dict_gpt2_original = gpt2_model_original.state_dict() # original(english)
state_dict_gpt2_ja = delete_transformer_prefixes_from_state_dict_keys(gpt2_model_ja.state_dict()) # # japanese
state_dict_gpt2_du = gpt2_model_du.state_dict() # dutch
state_dict_gpt2_ger = delete_transformer_prefixes_from_state_dict_keys(gpt2_model_ger.state_dict()) # german
state_dict_gpt2_ita = gpt2_model_ita.state_dict() # italian
state_dict_gpt2_fre = gpt2_model_fre.state_dict() # french
state_dict_gpt2_ko = delete_transformer_prefixes_from_state_dict_keys(gpt2_model_ko.state_dict()) # ko
state_dict_gpt2_spa = gpt2_model_spa.state_dict() # spanish
# ちゃんととれているか確認
# print(state_dict_gpt2_original.keys())
# state_dict_gpt2_ja.keys()
# state_dict_gpt2_du.keys()
# state_dict_gpt2_ger.keys()
# state_dict_gpt2_ita.keys()
# state_dict_gpt2_fre.keys()
# state_dict_gpt2_ko.keys()
# state_dict_gpt2_spa.keys()
# sys.exit()

""" 計算結果を保存する用の辞書 """

# 重みの比較結果を保存する辞書
weight_changes_gpt2 = defaultdict(defaultdict) # 全体の結果の保存先

# 個々のモデル・言語ペアの計算結果の保存先
weight_changes_gpt2_en_ja = defaultdict(dict)
weight_changes_gpt2_en_ger = defaultdict(dict)
weight_changes_gpt2_en_du = defaultdict(dict)
weight_changes_gpt2_en_ita = defaultdict(dict)
weight_changes_gpt2_en_fre = defaultdict(dict)
weight_changes_gpt2_en_ko = defaultdict(dict)
weight_changes_gpt2_en_spa = defaultdict(dict)

"""
各言語のモデルとoriginalモデル（英語）で、次元数が違う層があるかを確認: <- 結果、サイズの違うテンソルは全て埋め込み層のものと確認済み
"""
# print(f"en_ja: {is_different_in_terms_of_dim_size_of_each_layer(state_dict_gpt2_original, state_dict_gpt2_ja)}")
# print(f"en_du: {is_different_in_terms_of_dim_size_of_each_layer(state_dict_gpt2_original, state_dict_gpt2_du)}")
# print(f"en_ger: {is_different_in_terms_of_dim_size_of_each_layer(state_dict_gpt2_original, state_dict_gpt2_ger)}")
# print(f"en_ita: {is_different_in_terms_of_dim_size_of_each_layer(state_dict_gpt2_original, state_dict_gpt2_ita)}")
# print(f"en_fre: {is_different_in_terms_of_dim_size_of_each_layer(state_dict_gpt2_original, state_dict_gpt2_fre)}")
# print(f"en_ko: {is_different_in_terms_of_dim_size_of_each_layer(state_dict_gpt2_original, state_dict_gpt2_ko)}")

""" L1モデルとL1->L2モデル間の類似度/差異の計算 """

""" 英語->日本語 """
# 類似度などを計算
weight_changes_gpt2_en_ja_computed = compare_between_models(weight_changes_gpt2_en_ja, state_dict_gpt2_original, state_dict_gpt2_ja) # en_ja
# 辞書に結果を格納
weight_changes_gpt2['en_ja'] = weight_changes_gpt2_en_ja_computed

""" 英語->オランダ語 """
weight_changes_gpt2_en_du_computed = compare_between_models(weight_changes_gpt2_en_du, state_dict_gpt2_original, state_dict_gpt2_du) # en_du
weight_changes_gpt2['en_du'] = weight_changes_gpt2_en_du_computed

""" 英語->ドイツ語 """
weight_changes_gpt2_en_ger_computed = compare_between_models(weight_changes_gpt2_en_ger, state_dict_gpt2_original, state_dict_gpt2_ger) # en_du
weight_changes_gpt2['en_ger'] = weight_changes_gpt2_en_ger_computed

""" 英語->イタリア語 """
weight_changes_gpt2_en_ita_computed = compare_between_models(weight_changes_gpt2_en_ita, state_dict_gpt2_original, state_dict_gpt2_ita) # en_ita
weight_changes_gpt2['en_ita'] = weight_changes_gpt2_en_ita_computed

""" 英語->フランス語 """
weight_changes_gpt2_en_fre_computed = compare_between_models(weight_changes_gpt2_en_fre, state_dict_gpt2_original, state_dict_gpt2_fre) # en_fre
weight_changes_gpt2['en_fre'] = weight_changes_gpt2_en_fre_computed

""" 英語->韓国語 """
weight_changes_gpt2_en_ko_computed = compare_between_models(weight_changes_gpt2_en_ko, state_dict_gpt2_original, state_dict_gpt2_ko) # en_ko
weight_changes_gpt2['en_ko'] = weight_changes_gpt2_en_ko_computed

""" 英語->スペイン語 """
weight_changes_gpt2_en_spa_computed = compare_between_models(weight_changes_gpt2_en_spa, state_dict_gpt2_original, state_dict_gpt2_spa) # en_spa
weight_changes_gpt2['en_spa'] = weight_changes_gpt2_en_spa_computed

# 一応ちゃんと取れているか確認　
# print(weight_changes_gpt2.keys()) # dict_keys(['en_ja', 'en_du', 'en_ger', 'en_ita', 'en_fre', 'en_ko', 'en_spa'])
# print("\n")
# print("-----------------------------------------------------------------------")
# print("\n")
# print(weight_changes_gpt2.values())
# print("\n")
# print("-----------------------------------------------------------------------")
# print("\n")
# param_patterns = []
# for k in weight_changes_gpt2['en_ja'].keys():
#     param_patterns.append(k)
# print(param_patterns)
