"""
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
"""

import os
import sys
import dill as pickle
import random

import cld3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from funcs import (
    project_hidden_emb_to_vocab,
    get_hidden_states_with_edit_activation,
    layerwise_lang_stats,
    layerwise_lang_distribution,
    plot_lang_distribution,
    print_tokens,
    unfreeze_pickle,
)

""" prompts """
prompts = {
    "en": [
        "What are some popular tourist attractions in New York City?",
        "How can I improve my English writing skills?",
        "Can you recommend three must-read books from the science fiction genre?",
        "What are some effective strategies for time management?",
        "Where can I find authentic Italian cuisine in London?",
        "What are some tips for maintaining a healthy lifestyle?",
        "Can you suggest three classic movies from the 20th century?",
        "How can I develop good public speaking skills?",
        "What are some unique cultural traditions in Japan?",
        "Can you recommend three budget-friendly destinations for solo travelers?"
    ],
    "ja": [
        "ニューヨーク市で人気の観光名所はどこですか？",
        "英語のライティングスキルを向上させるにはどうすればいいですか？",
        "SFジャンルの必読書を3冊おすすめできますか？",
        "時間管理の効果的な戦略にはどのようなものがありますか？",
        "ロンドンで本格的なイタリア料理を食べられる場所はどこですか？",
        "健康的なライフスタイルを維持するためのヒントを教えてください。",
        "20世紀のクラシック映画を3本おすすめできますか？",
        "良いプレゼンテーションスキルを身につけるにはどうすればいいですか？",
        "日本のユニークな文化的伝統にはどのようなものがありますか？",
        "ソロ旅行者向けの予算に優しい旅行先を3つおすすめできますか？"
    ],
    "nl": [
        "Wat zijn enkele populaire toeristische attracties in New York City?",
        "Hoe kan ik mijn Engelse schrijfvaardigheid verbeteren?",
        "Kun je drie must-read boeken uit het sciencefictiongenre aanbevelen?",
        "Wat zijn enkele effectieve strategieën voor tijdmanagement?",
        "Waar kan ik authentieke Italiaanse gerechten vinden in Londen?",
        "Heb je tips voor een gezonde levensstijl?",
        "Kun je drie klassieke films uit de 20e eeuw aanbevelen?",
        "Hoe kan ik goede presentatievaardigheden ontwikkelen?",
        "Wat zijn enkele unieke culturele tradities in Japan?",
        "Kun je drie budgetvriendelijke bestemmingen voor soloreizigers aanbevelen?"
    ],
    "it": [
        "Quali sono alcune delle attrazioni turistiche più famose di New York City?",
        "Come posso migliorare le mie capacità di scrittura in inglese?",
        "Puoi consigliarmi tre libri imperdibili del genere fantascientifico?",
        "Quali sono alcune strategie efficaci per la gestione del tempo?",
        "Dove posso trovare cucina italiana autentica a Londra?",
        "Quali sono alcuni consigli per mantenere uno stile di vita sano?",
        "Puoi suggerire tre film classici del XX secolo?",
        "Come posso sviluppare buone capacità di parlare in pubblico?",
        "Quali sono alcune tradizioni culturali uniche in Giappone?",
        "Puoi consigliarmi tre destinazioni economiche per i viaggiatori solitari?"
    ],
    "ko": [
        "뉴욕에서 인기 있는 관광지는 어디인가요?",
        "영어 글쓰기 실력을 향상시키려면 어떻게 해야 하나요?",
        "SF 장르에서 꼭 읽어야 할 책 세 권을 추천해 줄 수 있나요?",
        "시간 관리를 효과적으로 하는 전략에는 어떤 것이 있나요?",
        "런던에서 정통 이탈리아 요리를 어디에서 먹을 수 있나요?",
        "건강한 생활 방식을 유지하는 팁이 있나요?",
        "20세기 클래식 영화 세 편을 추천해 줄 수 있나요?",
        "좋은 발표력을 기르려면 어떻게 해야 하나요?",
        "일본의 독특한 문화 전통에는 어떤 것이 있나요?",
        "혼자 여행하는 사람들에게 적합한 가성비 좋은 여행지 세 곳을 추천해 줄 수 있나요?"
    ]
}
prompts = {
    "en": "What are some popular tourist attractions in New York City?",
    "ja": "ニューヨーク市で人気の観光名所はどこですか？",
    "nl": "Wat zijn enkele populaire toeristische attracties in New York City?",
    "it": "Quali sono alcune delle attrazioni turistiche più famose di New York City?",
    "ko": "뉴욕에서 인기 있는 관광지는 어디인가요?"
}

""" model configs """
# LLaMA-3(8B)
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}

""" params """
device = "cuda" if torch.cuda.is_available() else "cpu"
layer_nums = 32
activation_types = ["abs", "product"]
norm_type = "no"
top_n = 20000
top_n_for_baseline = 50000
# L2 = "ja"

for activation_type in activation_types:
  for L2, model_name in model_names.items():
      model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
      tokenizer = AutoTokenizer.from_pretrained(model_name)

      """ get hidden states of all layers. """
      inputs = tokenizer(prompts[L2], return_tensors="pt").to(device)
      last_token_index = inputs["input_ids"].shape[1] - 1 # index corrensponding to the last token of the inputs.
      
      # run inference
      with torch.no_grad():
          output = model(**inputs, output_hidden_states=True)
      # ht
      all_hidden_states = output.hidden_states
      
      """ get topk tokens per each hidden state. """
      top_k = 100 # token nums to decode.
      # get hidden state of the layer(last token only).
      tokens_dict = project_hidden_emb_to_vocab(model, tokenizer, all_hidden_states, last_token_index, top_k=top_k)
      print_tokens(tokens_dict)
      # sys.exit()

      """ intervention with high AP neurons and baseline. """
      # get top AP neurons (layer_idx, neuron_idx)
      pkl_file_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/AUC/act_{activation_type}/ap_scores/{norm_type}_norm/sorted_neurons_{L2}.pkl"
      sorted_neurons_AP = unfreeze_pickle(pkl_file_path)[:top_n]
      # baseline
      sorted_neurons_AP_baseline = random.sample(sorted_neurons_AP[top_n_for_baseline+1:], len(sorted_neurons_AP[top_n_for_baseline+1:]))
      sorted_neurons_AP_baseline = sorted_neurons_AP_baseline[:top_n]

      # get hidden states with neurons intervention(high APs)
      all_hidden_states_high_AP_intervention = get_hidden_states_with_edit_activation(model, inputs, sorted_neurons_AP)
      # intervention (high APs.)
      tokens_dict_high_AP_intervention = project_hidden_emb_to_vocab(model, tokenizer, all_hidden_states_high_AP_intervention, last_token_index, top_k=top_k)
      print("============================================================== high APs ==============================================================\n")
      print_tokens(tokens_dict_high_AP_intervention)

      all_hidden_states_baseline_intervention = get_hidden_states_with_edit_activation(model, inputs, sorted_neurons_AP_baseline)
      tokens_dict_baseline_intervention = project_hidden_emb_to_vocab(model, tokenizer, all_hidden_states_baseline_intervention, last_token_index, top_k=top_k)
      print("============================================================== baseline ==============================================================\n")
      print_tokens(tokens_dict_baseline_intervention)

      """ visualization """
      if activation_type == "abs":
        # normal
        lang_stats = layerwise_lang_stats(tokens_dict, L2)
        lang_distribution = layerwise_lang_distribution(lang_stats, L2)
        plot_lang_distribution(lang_distribution, activation_type, "normal", top_n, L2)
      
      # high APs intervention
      lang_stats = layerwise_lang_stats(tokens_dict_high_AP_intervention, L2)
      lang_distribution = layerwise_lang_distribution(lang_stats, L2)
      plot_lang_distribution(lang_distribution, activation_type, "AP_intervention", top_n, L2)
      
      # baseline intervention
      lang_stats = layerwise_lang_stats(tokens_dict_baseline_intervention, L2)
      lang_distribution = layerwise_lang_distribution(lang_stats, L2)
      plot_lang_distribution(lang_distribution, activation_type, "baseline_intervention", top_n, L2)
      
      # delete caches
      del model
      torch.cuda.empty_cache()