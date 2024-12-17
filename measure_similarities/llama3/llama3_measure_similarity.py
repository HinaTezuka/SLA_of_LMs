"""
memo:
・　(llama3を使う場合は）hugging faceにログイン: !huggingface-cli login --token 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
・llama3(8B)を対象に調査
・google colabですでに同じ検証をしている: https://colab.research.google.com/drive/1h_wSGMWQRtcdFM88tgZdc3qLQjtDSN1u#scrollTo=8TbA3b1uqPeW
"""

import sys
sys.path.append('/home/s2410121/proj_LA/measure_similarities')
import similarity_funcs

import torch
import transformers

from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
# print('______module successfully loaded______')

""" モデル指定・ロード """

# モデル指定
llama_model_original_name = "meta-llama/Meta-Llama-3-8B" # en
# llama_model_ja_name = "lightblue/suzume-llama-3-8B-japanese" # ja
llama_model_ja_name = "tokyotech-llm/Llama-3-Swallow-8B-v0.1" # ja
llama_model_ger_name = "DiscoResearch/Llama3-German-8B" # ger
llama_model_ita_name = "DeepMount00/Llama-3-8b-Ita" # ita
llama_model_ko_name = "beomi/Llama-3-Open-Ko-8B" # korean
llama_model_ko_name = "beomi/Llama-3-KoEn-8B" # korean: enの基盤モデルにen, ko両方のテキストでkoreanをL2として追加学習済み
# llama_model_chi_name = "shareAI/llama3-Chinese-chat-8b" # chi : <- OSError: shareAI/llama3-Chinese-chat-8b does not appear to have a file named config.json. Checkout 'https://huggingface.co/shareAI/llama3-Chinese-chat-8b/tree/main' for available files.

# モデルのロード
llama_model_original = AutoModel.from_pretrained(llama_model_original_name)
llama_model_ja = AutoModel.from_pretrained(llama_model_ja_name)
llama_model_ger = AutoModel.from_pretrained(llama_model_ger_name)
llama_model_ita = AutoModel.from_pretrained(llama_model_ita_name)
llama_model_ko = AutoModel.from_pretrained(llama_model_ko_name)

""" それぞれのモデルのparametersを取得 """

# 重みを取得
state_dict_llama_original = llama_model_original.state_dict() # original(english)
state_dict_llama_ja = llama_model_ja.state_dict() # japanese
state_dict_llama_ger = llama_model_ger.state_dict() # german
state_dict_llama_ita = llama_model_ita.state_dict() # italian
state_dict_llama_ko = llama_model_ko.state_dict() # korean

# print(f'en_ger: {state_dict_llama_original.keys() == state_dict_llama_ger.keys()}')
# print(f'en_ja: {state_dict_llama_original.keys() == state_dict_llama_ja.keys()}')
# print(f'en_ita: {state_dict_llama_original.keys() == state_dict_llama_ita.keys()}')
# print(f'en_ko: {state_dict_llama_original.keys() == state_dict_llama_ko.keys()}')
# print(f'en: {len(state_dict_llama_original.keys())}, ja: {len(state_dict_llama_ja.keys())}, ger: {len(state_dict_llama_ger.keys())}, ita: {len(state_dict_llama_ita.keys())}, ko: {len(state_dict_llama_ko.keys())}')
# print(state_dict_llama_ger.keys())
# sys.exit()

# 重みの比較結果を保存する辞書
weight_changes_llama = {} # 全体の結果の保存先

# 個々のモデル・言語ペアの保存先
weight_changes_llama_en_ja = defaultdict(dict)
weight_changes_llama_en_ger = defaultdict(dict)
weight_changes_llama_en_ita = defaultdict(dict)
weight_changes_llama_en_ko = defaultdict(dict)

""" 類似度などを計算 """
""" 英語->日本語 """
# 類似度などを計算
weight_changes_llama_en_ja_computed = similarity_funcs.compare_between_models(weight_changes_llama_en_ja, state_dict_llama_original, state_dict_llama_ja) # en_ja
# 辞書に結果を格納
weight_changes_llama['en_ja'] = weight_changes_llama_en_ja_computed
""" 英語->ドイツ語 """
weight_changes_llama_en_ger_computed = similarity_funcs.compare_between_models(weight_changes_llama_en_ger, state_dict_llama_original, state_dict_llama_ger) # en_du
weight_changes_llama['en_ger'] = weight_changes_llama_en_ger_computed
""" 英語->イタリア語 """
weight_changes_llama_en_ita_computed = similarity_funcs.compare_between_models(weight_changes_llama_en_ita, state_dict_llama_original, state_dict_llama_ita) # en_ita
weight_changes_llama['en_ita'] = weight_changes_llama_en_ita_computed
""" 英語->韓国語 """
weight_changes_llama_en_ko_computed = similarity_funcs.compare_between_models(weight_changes_llama_en_ko, state_dict_llama_original, state_dict_llama_ko) # en_ko
weight_changes_llama['en_ko'] = weight_changes_llama_en_ita_computed

# 辞書に結果を格納
# weight_changes_llama['en_ja'] = weight_changes_llama_en_ja_computed
weight_changes_llama['en_ger'] = weight_changes_llama_en_ger_computed
weight_changes_llama['en_ita'] = weight_changes_llama_en_ita_computed
weight_changes_llama['en_ko'] = weight_changes_llama_en_ko_computed

# print(weight_changes_llama['en_ja'].keys())
