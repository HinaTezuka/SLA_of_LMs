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

import itertools
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from baukit import TraceDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

def print_accessible_modules(model) -> None:
  """
  check accessible modules (for baukit).
  """
  for name, module in model.named_modules():
    print(name)

def get_out_llama3_mlp(model, prompt, device):
  model.eval() # swith the model to evaluation mode (deactivate dropout, batch normalization)
  num_layers = model.config.num_hidden_layers  # nums of layers of the model
  MLP_values = [f"model.layers.{i}.mlp.down_proj" for i in range(num_layers)]  # generate path to MLP layer(of LLaMA-3)

  with torch.no_grad():
      # trace MLP layers using TraceDict
      with TraceDict(model, MLP_values) as ret:
          output = model(prompt, output_hidden_states=True, output_attentions=True)  # run inference
      MLP_value = [ret[act_value].output for act_value in MLP_values]  # get outputs of mlp per layer
      return MLP_value

def get_outputs_mlp(model, input_ids):
  MLP_values = get_out_llama3_mlp(model, input_ids, model.device)  # Llamaのself-att直後の値を取得
  # MLP_values = [act for act in MLP_values]
  MLP_values = [act[0].cpu() for act in MLP_values] # act[0]: tuple(attention_output, attention_weights, cache) <- act[0](attention_output)のみが欲しいのでそれをcpu上に配置

  return MLP_values

def get_similarities_mlp(model, tokenizer, data) -> defaultdict(list):
  """
  get cosine similarities of MLP block outputs of all the sentences in data (for same semantics or non-same semantics pairs)

  block: "mlp"

  return: defaultdict(list):
      {
        layer_idx: [] <- list of similarities per a pair
      }
  """
  similarities = defaultdict(list)
  for L1_text, L2_text in data:
    input_ids_L1 = tokenizer(L1_text, return_tensors="pt").input_ids.to(device)
    token_len_L1 = len(input_ids_L1[0])
    input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to(device)
    token_len_L2 = len(input_ids_L2[0])

    output_L1 = get_outputs_mlp(model, input_ids_L1)
    output_L2 = get_outputs_mlp(model, input_ids_L2)
    """
    shape of outputs: 32(layer_num) lengthのリスト。
    outputs[0]: 0層目のmlpを通った後のrepresentation
    """
    # mlpの出力を取得
    for layer_idx in range(32):
      """
      各レイヤーの最後のトークンに対応するmlp_outputを取得（output_L1[layer_idx][0][token_len_L1-1]) + 2次元にreshape(2次元にreshapeしないとcos_simが測れないため。)
      """
      similarity = cosine_similarity(output_L1[layer_idx][token_len_L1-1].unsqueeze(0), output_L2[layer_idx][token_len_L2-1].unsqueeze(0))
      similarities[layer_idx].append(similarity[0][0]) # for instance, similarity=[[0.93852615]], so remove [[]] and extract similarity value only

  return similarities

# cosine similarity
def plot_hist(dict1: defaultdict(float), dict2: defaultdict(float), L2: str) -> None:
    # convert keys and values into list
    keys = np.array(list(dict1.keys()))
    values1 = list(dict1.values())
    values2 = list(dict2.values())

    offset = 0.1 # バーをずらす用

    # plot hist
    plt.bar(keys-offset, values1, alpha=1, label='same semantics')
    plt.bar(keys+offset, values2, alpha=1, label='different semantics')

    plt.xlabel('Layer index', fontsize=35)
    plt.ylabel('Cosine Sim', fontsize=35)
    plt.title(f'en_{L2}')
    plt.tick_params(axis='x', labelsize=15)  # x軸の目盛りフォントサイズ
    plt.tick_params(axis='y', labelsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/s2410121/proj_LA/measure_similarities/llama3/images/mlp_outputs_sim/cos_sim/en_{L2}_revised.png",
                bbox_inches="tight",
            )
    plt.close()


if __name__ == "__main__":
  """ model configs """
  # LLaMA-3
  model_names = {
      # "base": "meta-llama/Meta-Llama-3-8B",
      "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
      # "de": "DiscoResearch/Llama3-German-8B", # ger
      "nl": "ReBatch/Llama-3-8B-dutch", # du
      "it": "DeepMount00/Llama-3-8b-Ita", # ita
      "ko": "beomi/Llama-3-KoEn-8B", # ko
  }

  for L2, model_name in model_names.items():
    L1 = "en" # L1 is fixed to english.

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    """ tatoeba translation corpus """
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # select first 2000 sentences
    total_sentence_num = 2000 if L2 == "ko" else 5000
    num_sentences = 2000
    dataset = dataset.select(range(total_sentence_num))
    tatoeba_data = []
    for sentence_idx, item in enumerate(dataset):
        if sentence_idx == num_sentences: break
        # check if there are empty sentences.
        if item['translation'][L1] != '' and item['translation'][L2] != '':
            tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
    tatoeba_data_len = len(tatoeba_data)

    """
    baseとして、対訳関係のない1文ずつのペアを作成
    """
    random_data = []
    if L2 == "ko": # koreanはデータ数が足りない
        dataset2 = load_dataset("tatoeba", lang1=L1, lang2="ja", split="train").select(range(5000))
    for sentence_idx, item in enumerate(dataset):
        if sentence_idx == num_sentences: break
        if L2 == "ko" and dataset2['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
            random_data.append((dataset2["translation"][num_sentences+sentence_idx][L1], item["translation"][L2])) 
        elif L2 != "ko" and dataset['translation'][num_sentences+sentence_idx][L1] != '' and item['translation'][L2] != '':
            random_data.append((dataset["translation"][num_sentences+sentence_idx][L1], item["translation"][L2]))

    print(f'non-translation pair: {random_data[-10:]}')

    """ calc similarities """
    results_same_semantics = get_similarities_mlp(model, tokenizer, tatoeba_data)
    results_non_same_semantics = get_similarities_mlp(model, tokenizer, random_data)
    final_results_same_semantics = defaultdict(float)
    final_results_non_same_semantics = defaultdict(float)
    for layer_idx in range(32): # ３２層
        final_results_same_semantics[layer_idx] = np.array(results_same_semantics[layer_idx]).mean()
        final_results_non_same_semantics[layer_idx] = np.array(results_non_same_semantics[layer_idx]).mean()


    # delete some cache
    del model
    torch.cuda.empty_cache()

    """ plot """
    plot_hist(final_results_same_semantics, final_results_non_same_semantics, L2)

  print("visualization completed !")
