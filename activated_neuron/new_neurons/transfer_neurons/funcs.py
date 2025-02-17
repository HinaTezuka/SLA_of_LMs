"""
neuron detection for MLP Block of LLaMA-3(8B).

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

MistralForCausalLM(
  (model): MistralModel(
    (embed_tokens): Embedding(32768, 4096)
    (layers): ModuleList(
      (0-31): 32 x MistralDecoderLayer(
        (self_attn): MistralSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): MistralRotaryEmbedding()
        )
        (mlp): MistralMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): MistralRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): MistralRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): MistralRMSNorm((4096,), eps=1e-05)
  )
  (lm_head): Linear(in_features=4096, out_features=32768, bias=False)
)
"""
import os
import sys
import pickle
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from baukit import TraceDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

def monolingual_dataset(lang: str, num_sentences: int) -> list:
    """
    making tatoeba translation corpus for lang-specific neuron detection.
    """

    tatoeba_data = []
    dataset = load_dataset("tatoeba", lang1="en", lang2=lang, split="train")
    dataset = dataset.select(range(2500))
    random.seed(42)
    random_indices = random.sample(range(2500), num_sentences)
    dataset = dataset.select(random_indices)
    for sentence_idx, item in enumerate(dataset):
        tatoeba_data.append(item['translation'][lang])
    
    return tatoeba_data

def multilingual_dataset_for_lang_specific_detection(langs: list, num_sentences=500) -> list:
    """
    making tatoeba translation corpus for lang-specific neuron detection.
    """

    tatoeba_data = []
    for lang in langs:
        if "en" in langs and lang == "en":
            dataset = load_dataset("tatoeba", lang1="en", lang2="ja", split="train")
            # select random num_sentences samples from top 2500.
            dataset = dataset.select(range(2500))
            random.seed(42)
            random_indices = random.sample(range(2500), num_sentences)
            dataset = dataset.select(random_indices)
            for item in dataset:
                tatoeba_data.append(item['translation'][lang])
            continue
        dataset = load_dataset("tatoeba", lang1="en", lang2=lang, split="train")
        dataset = dataset.select(range(2500))
        random.seed(42)
        random_indices = random.sample(range(2500), num_sentences)
        dataset = dataset.select(random_indices)
        for sentence_idx, item in enumerate(dataset):
            tatoeba_data.append(item['translation'][lang])
    
    return tatoeba_data

def multilingual_dataset_for_centroid_detection(langs: list, num_sentences=500) -> list:
    """
    making tatoeba translation corpus for lang-specific neuron detection.
    return_dict(tatoeba_data):
    {
        lang: [text1, text2, ...]
    }
    """

    tatoeba_data = defaultdict(list)
    for lang in langs:
        dataset = load_dataset("tatoeba", lang1="en", lang2=lang, split="train")
        dataset = dataset.select(range(2500))
        random.seed(42)
        random_indices = random.sample(range(2500), num_sentences)
        dataset = dataset.select(random_indices)
        for sentence_idx, item in enumerate(dataset):
            en_txt = item['translation']['en']
            l2_txt = item['translation'][lang]
            if en_txt != '' and l2_txt != '':
                tatoeba_data[lang].append((en_txt, l2_txt))
    
    return dict(tatoeba_data)

def get_out_llama3_act_fn(model, prompt, device, index):
    model.eval() # swith the model to evaluation mode (deactivate dropout, batch normalization)
    num_layers = model.config.num_hidden_layers  # nums of layers of the model
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(num_layers)]  # generate path to MLP layer(of LLaMA-3)

    with torch.no_grad():
        # trace MLP layers using TraceDict
        with TraceDict(model, MLP_act) as ret:
            output = model(prompt, output_hidden_states=True, output_attentions=True)  # モデルを実行
        MLP_act_value = [ret[act_value].output for act_value in MLP_act]  # 各MLP層の出力を取得
        return MLP_act_value

def get_out_llama3_up_proj(model, prompt, device, index):
    model.eval() # swith the model to evaluation mode (deactivate dropout, batch normalization)
    num_layers = model.config.num_hidden_layers  # nums of layers of the model
    MLP_act = [f"model.layers.{i}.mlp.up_proj" for i in range(num_layers)]  # generate path to MLP layer(of LLaMA-3)

    with torch.no_grad():
        # trace MLP layers using TraceDict
        with TraceDict(model, MLP_act) as ret:
            output = model(prompt, output_hidden_states=True, output_attentions=True)
        MLP_act_value = [ret[act_value].output for act_value in MLP_act]
        return MLP_act_value

def act_llama3(model, input_ids):
    act_fn_values = get_out_llama3_act_fn(model, input_ids, model.device, -1)  # LlamaのMLP活性化を取得
    act_fn_values = [act.to("cpu") for act in act_fn_values] # Numpy配列はCPUでしか動かないので、各テンソルをCPU上へ移動
    up_proj_values = get_out_llama3_up_proj(model, input_ids, model.device, -1)
    up_proj_values = [act.to("cpu") for act in up_proj_values]

    return act_fn_values, up_proj_values

# act_fn(x)とup_proj(x)の要素積を計算(=neuronとする)
def calc_element_wise_product(act_fn_value, up_proj_value):
    return act_fn_value * up_proj_value

def track_neurons_with_text_data(model, device, tokenizer, data, start_idx, end_idx, is_last_token_only=False, is_bilingual=False):
    # layers_num
    num_layers = 32
    # nums of total neurons (per a layer)
    num_neurons = 14336
    # return dict for saving activation values.
    # activation_dict = defaultdict(list)
    activation_dict = {}
    labels = []
    """
    activation_dict
    {
    (layer_idx, neuron_idx): [act_value1, act_value2, ...]
    }
    labels: [1, 1, ..., 0, 0, ...]
    """

    # Track neurons with tatoeba
    for text_idx, text in enumerate(data):
        """
        get activation values
        mlp_activation: [torch.Tensor(batch_size, sequence_length, num_neurons) * num_layers]
        """
        # input text
        inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)
        token_len = len(inputs[0])
        act_fn_values, up_proj_values = act_llama3(model, inputs)
        """
        neurons(in llama3 MLP): up_proj(x) * act_fn(gate_proj(x)) <- input of down_proj()
        """
        for layer_idx in range(num_layers):
            # consider means of all token activations.
            if not is_last_token_only:
                """ consider average of all tokens """
                act_values_per_token = []
                for token_idx in range(token_len):
                    act_fn_value = act_fn_values[layer_idx][:, token_idx, :][0]
                    up_proj_value = up_proj_values[layer_idx][:, token_idx, :][0]

                    """ calc and extract neurons: up_proj(x) * act_fn(x) """
                    act_values = calc_element_wise_product(act_fn_value, up_proj_value)
                    act_values_per_token.append(act_values)
                """ calc average act_values of all tokens """
                act_values_all_token = np.array(act_values_per_token)
                means = np.mean(act_values_all_token, axis=0)
            # consider last token activations only.
            elif is_last_token_only:
                act_fn_value = act_fn_values[layer_idx][:, token_len-1, :][0]
                up_proj_value = up_proj_values[layer_idx][:, token_len-1, :][0]
                means = calc_element_wise_product(act_fn_value, up_proj_value)
            # save in activation_dict
            for neuron_idx in range(num_neurons):
                mean_act_value = means[neuron_idx]
                # activation_dict[(layer_idx, neuron_idx)].append(mean_act_value)
                activation_dict.setdefault((layer_idx, neuron_idx), []).append(mean_act_value)

        # labels list
        if not is_bilingual:
            if text_idx >= start_idx and text_idx < end_idx:
                labels.append(1)
            else:
                labels.append(0)
        elif is_bilingual:
            if (text_idx >= start_idx and text_idx < end_idx) or (text_idx >= 400 and text_idx < 500):
                labels.append(1)
            else:
                labels.append(0)            

    return activation_dict, labels

def get_hidden_states(model, tokenizer, device, num_layers, data):
    """
    """
    # { layer_idx: [c_1, c_2, ...]} c_1: (last token)centroid of text1 (en-L2).
    c_hidden_states = defaultdict(list)

    for text1, text2 in data:
        inputs1 = tokenizer(text1, return_tensors="pt").to(device) # english text
        inputs2 = tokenizer(text2, return_tensors="pt").to(device) # L2 text

        # get hidden_states
        with torch.no_grad():
            output1 = model(**inputs1, output_hidden_states=True)
            output2 = model(**inputs2, output_hidden_states=True)

        all_hidden_states1 = output1.hidden_states[1:] # remove embedding layer
        all_hidden_states2 = output2.hidden_states[1:]
        last_token_index1 = inputs1["input_ids"].shape[1] - 1
        last_token_index2 = inputs2["input_ids"].shape[1] - 1

        """  """
        for layer_idx in range(num_layers):
            hs1 = all_hidden_states1[layer_idx][:, last_token_index1, :].squeeze().detach().cpu().numpy()
            hs2 = all_hidden_states2[layer_idx][:, last_token_index2, :].squeeze().detach().cpu().numpy()
            # save mean of (en_ht, L2_ht). <- estimated shared point in shared semantic space.
            c = np.stack([hs1, hs2])
            c = np.mean(c, axis=0)
            c_hidden_states[layer_idx].append(c)

    return dict(c_hidden_states)

def get_centroid_of_shared_space(hidden_states: dict):
    centroids = [] # [c1, c2, ] len = layer_num(32layers: 0-31)
    for layer_idx, c in hidden_states.items():
        final_c = np.mean(c, axis=0) # calc mean of c(shared point per text) of all text.
        centroids.append(final_c)
    
    return centroids

# def get_centroids_per_L2(hidden_states: defaultdict(list)):
#     centroids = [] # [c1, c2, ...] c1: centroid of layer1.
#     for layer_idx, hidden_states_layer in hidden_states.items():
#         hidden_states_layer = np.array(hidden_states_layer)
#         centroids.append(np.mean(hidden_states_layer, axis=0))
    
#     return centroids

# def get_centroids_of_shared_space(centroids: dict, num_layers: int):
#     # c_langs = centroids.values()
#     # print(len(c_langs))
#     c_ja = centroids["ja"]
#     c_nl = centroids["nl"]
#     c_ko = centroids["ko"]
#     c_it = centroids["it"]
#     stacked_centroids_all_langs = np.stack([c_ja, c_nl, c_ko, c_it])

#     c_shared_space = np.mean(stacked_centroids_all_langs, axis=0)

#     return c_shared_space

def get_out_llama3_post_attention_layernorm(model, prompt, device, index):
    model.eval() # swith the model to evaluation mode (deactivate dropout, batch normalization)
    num_layers = model.config.num_hidden_layers  # nums of layers of the model
    ATT_act = [f"model.layers.{i}.post_attention_layernorm" for i in range(num_layers)]  # generate path to MLP layer(of LLaMA-3)

    with torch.no_grad():
        # trace MLP layers using TraceDict
        with TraceDict(model, ATT_act) as ret:
            output = model(prompt, output_hidden_states=True, output_attentions=True)
        ATT_act_value = [ret[act_value].output for act_value in ATT_act]
        return ATT_act_value

def post_attention_llama3(model, input_ids):
    values = get_out_llama3_post_attention_layernorm(model, input_ids, model.device, -1)
    values = [value.to("cpu") for value in values]

    return values

def compute_scores(model, tokenizer, device, data, neurons, centroids):
    """
    data: texts in specific L2 ([txt1, txt2, ...]).
    neurons: [(l, i), (l, i), ...] l: layer_idx, i: neuron_idx.
    """
    num_layers = model.config.num_hidden_layers
    scores = defaultdict(list) # { (l, i): [score1, score2, ...] }
    for text in data:
        inputs_for_ht = tokenizer(text, return_tensors="pt").to(device)
        inputs_for_att = inputs_for_ht.input_ids

        token_len = len(inputs_for_att[0])
        last_token_idx = token_len-1
        # get ht
        hts = [] # hidden_states: [ht_layer1, ht_layer2, ...]
        atts = [] # post_att_LN_outputs: [atts_layer1, atts_layer2, ...]
        acts = [] # activation_values: [acts_layer1, acts_layer2, ...]
        with torch.no_grad():
            outputs = model(**inputs_for_ht, output_hidden_states=True)
        ht_all_layer = outputs.hidden_states[1:]
        # get representation(right after post_att_LN).
        post_attention_layernorm_values = post_attention_llama3(model, inputs_for_att)
        # get activation_values(in MLP).
        act_fn_values, up_proj_values = act_llama3(model, inputs_for_att)

        for layer_idx in range(num_layers):
            # hidden states
            hts.append(ht_all_layer[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy())
            # post attention
            atts.append(post_attention_layernorm_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy())
            # act_value * up_proj
            act_fn_value = act_fn_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            up_proj_value = up_proj_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            act_values = calc_element_wise_product(act_fn_value, up_proj_value)
            acts.append(act_values)
        
        print(atts[0])
        print(hts[0])
        print(acts[0])
        sys.exit()

def save_as_pickle(file_path: str, target_dict) -> None:
    """
    Save a dictionary as a pickle file with improved safety.
    """
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temp_path = file_path + ".tmp"  # Temporary file for safe writing

    try:
        # Write to a temporary file
        with open(temp_path, "wb") as f:
            pickle.dump(target_dict, f)
        # Replace the original file with the temporary file
        os.replace(temp_path, file_path)
        print("pkl_file successfully saved.")
    except Exception as e:
        # Clean up temporary file if an error occurs
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e  # Re-raise the exception for further handling

def unfreeze_pickle(file_path: str):
    """
    Load a pickle file as a dictionary with error handling.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Error unpickling file {file_path}: {e}")

def compute_ap_and_sort(activations_dict: dict, labels: list):
    """
    calc APscore and sort (considering nums_of_label) for detecting L2-specific neurons.

    Args:
        activations: activation_value for each sentences.
        start_label1: indicating idx where label1(corresponding to L2) sentences bigin.

    Returns:
        sorted_neurons list, ニューロンごとのAPスコア辞書
    """

    # calc AP score for each shared neuron and calc total score which consider nums of label
    final_scores = {}
    for (layer_idx, neuron_idx), activations in activations_dict.items():
        # calc AP score
        ap = average_precision_score(y_true=labels, y_score=activations)
        # save total score
        final_scores[(layer_idx, neuron_idx)] = ap

    # sort: based on total score of each neuron
    sorted_neurons = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)

    return sorted_neurons, final_scores

# visualization
def plot_hist(dict1: defaultdict(float), dict2: defaultdict(float), L2: str, AUC_or_AUC_baseline:str, intervention_num: str) -> None:
    # convert keys and values into list
    keys = np.array(list(dict1.keys()))
    values1 = list(dict1.values())
    values2 = list(dict2.values())

    offset = 0.1 # バーをずらす用

    # plot hist
    plt.bar(keys-offset, values1, alpha=1, label='same semantics')
    plt.bar(keys+offset, values2, alpha=1, label='different semantics')
    # plt.bar(keys, values1, alpha=1, label='same semantics')
    # plt.bar(keys, values2, alpha=1, label='different semantics')

    plt.xlabel('Layer index', fontsize=35)
    plt.ylabel('Cosine Sim', fontsize=35)
    plt.title(f'en_{L2}')
    plt.tick_params(axis='x', labelsize=15)  # x軸の目盛りフォントサイズ
    plt.tick_params(axis='y', labelsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/llama3/{L2}.png",
        bbox_inches="tight"
    )
    plt.close()

""" func for editing activation values """
def edit_activation(output, layer, layer_idx_and_neuron_idx):
    """
    edit activation value of neurons(indexed layer_idx and neuron_idx)
    output: activation values
    layer: sth like 'model.layers.{layer_idx}.mlp.act_fn'
    layer_idx_and_neuron_idx: list of tuples like [(layer_idx, neuron_idx), ....]
    """
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if str(layer_idx) in layer:  # layer名にlayer_idxが含まれているか確認
            output[:, :, neuron_idx] *= 0  # 指定されたニューロンの活性化値をゼロに設定

    return output

def take_similarities_with_edit_activation(model, tokenizer, device, layer_neuron_list, data):
    trace_layers = [f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:

        return calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, device, data)

def calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, device, data):
    """
    各層について、2000文ペアそれぞれのhidden_statesの類似度の平均を計算
    """
    similarities = defaultdict(list) # {layer_idx: mean_sim_of_each_sentences}

    for L1_txt, L2_txt in data:
        hidden_states = defaultdict(torch.Tensor)
        inputs_L1 = tokenizer(L1_txt, return_tensors="pt").to(device)
        inputs_L2 = tokenizer(L2_txt, return_tensors="pt").to(device)

        # get hidden_states
        with torch.no_grad():
            output_L1 = model(**inputs_L1, output_hidden_states=True)
            output_L2 = model(**inputs_L2, output_hidden_states=True)

        all_hidden_states_L1 = output_L1.hidden_states
        all_hidden_states_L2 = output_L2.hidden_states
        # 最後のtokenのhidden_statesのみ取得
        last_token_index_L1 = inputs_L1["input_ids"].shape[1] - 1
        last_token_index_L2 = inputs_L2["input_ids"].shape[1] - 1

        """各層の最後のトークンの hidden state をリストに格納 + 正規化 """
        last_token_hidden_states_L1 = [
            (layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy() /
            np.linalg.norm(layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy(), axis=-1, keepdims=True))
            for layer_hidden_state in all_hidden_states_L1
        ]
        last_token_hidden_states_L2 = [
            (layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy() /
            np.linalg.norm(layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy(), axis=-1, keepdims=True))
            for layer_hidden_state in all_hidden_states_L2
        ]
        # cos_sim
        similarities = calc_cosine_sim(last_token_hidden_states_L1, last_token_hidden_states_L2, similarities)

    return similarities

def calc_cosine_sim(last_token_hidden_states_L1: list, last_token_hidden_states_L2: list, similarities: defaultdict(float)) -> defaultdict(float):
    """
    層ごとの類似度を計算
    """
    for layer_idx, (hidden_state_L1, hidden_state_L2) in enumerate(zip(last_token_hidden_states_L1, last_token_hidden_states_L2)):
        sim = cosine_similarity(hidden_state_L1, hidden_state_L2)[0, 0] # <- [[0.50695133]] のようになっているので、数値の部分だけ抽出
        similarities[layer_idx].append(sim)

    return similarities