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
import torch.nn.functional as F
from baukit import TraceDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

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

def monolingual_dataset_en(num_sentences: int) -> list:
    """
    making tatoeba translation corpus for lang-specific neuron detection.
    """
    tatoeba_data = []
    dataset = load_dataset("tatoeba", lang1="en", lang2="nl", split="train")
    dataset = dataset.select(range(2500))
    random.seed(42)
    random_indices = random.sample(range(2500), num_sentences)
    dataset = dataset.select(random_indices)
    for item in dataset:
        tatoeba_data.append(item['translation']["en"])
    
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

def sentence_pairs_for_train_test_split(L2: str, num_sentences=2000) -> list:
    """
    """
    tatoeba_data = []
    dataset = load_dataset("tatoeba", lang1="en", lang2=L2, split="train")
    dataset = dataset.select(range(2500))
    random.seed(42)
    random_indices = random.sample(range(2500), num_sentences)
    dataset = dataset.select(random_indices)
    for sentence_idx, item in enumerate(dataset):
        en_txt = item['translation']['en']
        l2_txt = item['translation'][L2]
        if en_txt != '' and l2_txt != '':
            tatoeba_data.append((en_txt, l2_txt))
    
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

def get_norms_of_value_vector(model, tokenizer, device: str, neurons: list, data: list) -> float:
    result = defaultdict(list)
    for txt in data:
        inputs = tokenizer(txt, return_tensors="pt").input_ids.to(device)
        token_len = len(inputs[0])
        act_fn_values, up_proj_values = act_llama3(model, inputs)
        for layer_idx, neuron_idx in neurons:
            act_fn_value = act_fn_values[layer_idx][:, token_len-1, :][0]
            up_proj_value = up_proj_values[layer_idx][:, token_len-1, :][0]
            act_values = calc_element_wise_product(act_fn_value, up_proj_value).detach().cpu().numpy()
            act_value = act_values[neuron_idx]
            value_vector = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[neuron_idx].detach().cpu().numpy()
            norm = np.linalg.norm(act_value * value_vector)
            result[(layer_idx, neuron_idx)].append(norm)
    # calc means.
    for layer_idx, neuron_idx in neurons:
        result[(layer_idx, neuron_idx)] = np.mean(np.array(result[(layer_idx, neuron_idx)]))

    return dict(result)

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

def get_activations_hook(model, input, layer_idx: int, text_idx, activation_array: np.array):
    activation_values = input[0][0, -1, :].detach().cpu().numpy()
    activation_array[layer_idx, :, text_idx] = activation_values

def track_neurons_with_text_data(model, model_type, device, tokenizer, data, start_idx, end_idx):
    num_layers = 32 if model_type != 'bloom' else 30
    num_neurons = 14336 if model_type != 'bloom' else 10240

    # a numpy array for saving activation values (layer_idx * neuron_idx * text_idx):
    activation_array = np.zeros((num_layers, num_neurons, len(data)), dtype=np.float16)
    labels = [] # labels: [1, 1, ..., 0, 0, ...] <- 1 for the texts of target language, 0 for others.

    # Track neurons with tatoeba
    for text_idx, text in enumerate(data):
        inputs = tokenizer(text, return_tensors='pt').input_ids.to(device)
        token_len = len(inputs[0])

        # set hooks.
        handles = []
        if model_type in ['llama3', 'mistral', 'aya']:
            for layer_idx, layer in enumerate(model.model.layers):
                handle = layer.mlp.down_proj.register_forward_pre_hook(
                    lambda model, input, layer_idx=layer_idx, text_idx=text_idx, activation_array=activation_array: get_activations_hook(model, input, layer_idx, text_idx, activation_array)
                )
                handles.append(handle)
        elif model_type in ['bloom']:
            for layer_idx, layer in enumerate(model.transformer.h):
                handle = layer.mlp.dense_4h_to_h.register_forward_pre_hook(
                    lambda model, input, layer_idx=layer_idx, text_idx=text_idx, activation_array=activation_array: get_activations_hook(model, input, layer_idx, text_idx, activation_array)
                )
                handles.append(handle)

        # run inference.
        with torch.no_grad():
            output = model(inputs)

        # remove handles.
        for handle in handles:
            handle.remove()

        # Assign labels
        if start_idx <= text_idx < end_idx:
            labels.append(1)
        else:
            labels.append(0)

    return activation_array, labels

def track_neurons_with_text_data_elem_wise(model, device, tokenizer, qa, qa_num, L2): # for qa dataset.
    # layers_num
    num_layers = 32
    # nums of total neurons (per a layer)
    num_neurons = 14336
    # returning np_array.
    activation_array = np.zeros((num_layers, num_neurons, qa_num))
    """ """
    for i in range(len(qa['queries'])):
        if i == qa_num: break # 1000 questions.
        """ make prompt and input_ids. """
        q = qa['queries'][i][L2] # question
        a = qa['answers'][i][L2][0]['text'] # answer
        if q == '' or q == None or  a == '' or a == None:
            continue
        ans_patterns = {
        'ja': '答え: ',
        'nl': 'Antwoord: ',
        'ko': '답변: ',
        'it': 'Risposta: ',
        }
        prompt = f'{q}? {ans_patterns[L2]}'
        # input text
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        token_len = len(inputs[0])

        """ get activations. """
        act_values = [] # len: layer_num
        # hook fn
        def get_elem_wise_product(model, input):
            act_values.append(input[0][0][-1].detach().cpu().numpy()) # last tokenに対応する活性値のみ取得
        handles = []
        for layer in model.model.layers:
            handle = layer.mlp.down_proj.register_forward_pre_hook(get_elem_wise_product)
            handles.append(handle)
        # run inference.
        with torch.no_grad():
            output = model(inputs)
        # remove hook
        for handle in handles:
            handle.remove()

        for layer_idx in range(num_layers):
            for neuron_idx in range(num_neurons):
                activation_array[layer_idx, neuron_idx, i] = act_values[layer_idx][neuron_idx] # i: question_idx

    return activation_array

def get_hidden_states_including_emb_layer_with_edit_activation(model, model_type, tokenizer, device, layer_neuron_list, num_layers, data):
    if model_type == 'bloom':
        trace_layers = list(set([f'transformer.h.{layer}.mlp.gelu_impl' for layer, _ in layer_neuron_list]))
    else:
        trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]))
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:

        return get_hidden_states_including_emb_layer(model, tokenizer, device, num_layers, data)

def get_hidden_states_including_emb_layer(model, tokenizer, device, num_layers, data):
    """
    """
    # { layer_idx: [c_1, c_2, ...]} c_1: (last token)centroid of text1 (en).
    c_hidden_states = defaultdict(list)

    torch.manual_seed(42)
    for text1 in data:
        inputs1 = tokenizer(text1, return_tensors="pt").to(device) # english text

        with torch.no_grad():
            output1 = model(**inputs1, output_hidden_states=True)

        all_hidden_states1 = output1.hidden_states # include embedding layer
        last_token_index1 = inputs1["input_ids"].shape[1] - 1

        for layer_idx in range(len(all_hidden_states1)):
            hs1 = all_hidden_states1[layer_idx][:, last_token_index1, :].squeeze().detach().cpu().numpy()
            c_hidden_states[layer_idx].append(hs1)

    return dict(c_hidden_states)

def get_hidden_states(model, tokenizer, device, num_layers, data):
    """
    """
    # { layer_idx: [c_1, c_2, ...]} c_1: (last token)centroid of text1 (en-L2).
    c_hidden_states = defaultdict(list)

    torch.manual_seed(42)
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

def get_hidden_states_en_only(model, tokenizer, device, num_layers, data):
    """
    """
    # { layer_idx: [c_1, c_2, ...]} c_1: (last token)centroid of text1 (en).
    c_hidden_states = defaultdict(list)

    torch.manual_seed(42)
    for text1 in data:
        inputs1 = tokenizer(text1, return_tensors="pt").to(device) # english text

        # get hidden_states
        with torch.no_grad():
            output1 = model(**inputs1, output_hidden_states=True)

        all_hidden_states1 = output1.hidden_states[1:] # exclude embedding layer
        last_token_index1 = inputs1["input_ids"].shape[1] - 1

        """  """
        for layer_idx in range(num_layers):
            hs1 = all_hidden_states1[layer_idx][:, last_token_index1, :].squeeze().detach().cpu().numpy()
            c_hidden_states[layer_idx].append(hs1)

    return dict(c_hidden_states)

def get_centroid_of_shared_space(hidden_states: dict):
    centroids = [] # [c1, c2, ] len = layer_num
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

def get_all_outputs_llama3_mistral(model, prompt, device):
    model.eval()
    num_layers = model.config.num_hidden_layers
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(num_layers)]
    MLP_up_proj = [f"model.layers.{i}.mlp.up_proj" for i in range(num_layers)]
    ATT_act = [f"model.layers.{i}.self_attn.o_proj" for i in range(num_layers)]
    # ATT_act = [f"model.layers.{i}.post_attention_layernorm" for i in range(num_layers)]

    torch.manual_seed(42)
    with TraceDict(model, MLP_act + MLP_up_proj + ATT_act) as ret:
        with torch.no_grad():
            outputs = model(prompt, output_hidden_states=True, output_attentions=True)
    
    MLP_act_values = [ret[act].output for act in MLP_act]
    up_proj_values = [ret[proj].output for proj in MLP_up_proj]
    ATT_values = [ret[att].output for att in ATT_act]
    
    return MLP_act_values, up_proj_values, ATT_values, outputs

def get_all_outputs_llama3_mistral(model, prompt, device):
    model.eval()
    num_layers = model.config.num_hidden_layers
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(num_layers)]
    MLP_up_proj = [f"model.layers.{i}.mlp.up_proj" for i in range(num_layers)]
    ATT_act = [f"model.layers.{i}.self_attn.o_proj" for i in range(num_layers)]
    # ATT_act = [f"model.layers.{i}.post_attention_layernorm" for i in range(num_layers)]

    torch.manual_seed(42)
    with TraceDict(model, MLP_act + MLP_up_proj + ATT_act) as ret:
        with torch.no_grad():
            outputs = model(prompt, output_hidden_states=True, output_attentions=True)
    
    MLP_act_values = [ret[act].output for act in MLP_act]
    up_proj_values = [ret[proj].output for proj in MLP_up_proj]
    ATT_values = [ret[att].output for att in ATT_act]
    
    return MLP_act_values, up_proj_values, ATT_values, outputs

def compute_scores_optimized(model, tokenizer, device, data, candidate_neurons, centroids, score_type):
    """
    Optimized function to compute scores using batched inference.
    """
    num_layers = model.config.num_hidden_layers
    num_candidate_layers = len(candidate_neurons.keys())
    num_neurons = 14336
    final_scores_save = np.zeros((num_candidate_layers, num_neurons, len(data))) # final_scoreを計算する前の保存用
    final_scores = np.zeros((num_candidate_layers, num_neurons))

    for text_idx, text in enumerate(data):
        # Tokenize all input data at once
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Perform the forward pass once to get all necessary values
        MLP_act_values, up_proj_values, post_attention_values, outputs = get_all_outputs_llama3_mistral(model, inputs.input_ids, device)

        # Extract hidden states (ht), attention layer outputs, and MLP activations
        ht_all_layer = outputs.hidden_states # len==33(0-32), including 0th layer(embedding layer). <- emb(0th) layer is needed to calc score of 1th layer.
        token_len = inputs.input_ids.size(1)
        last_token_idx = token_len - 1

        # Score calculation based on type (L2 distance or cosine similarity)
        # c = np.mean(centroids[5:21], axis=0).reshape(1, -1) # centroids: mean of c for 5-20layers.
        for layer_idx, neurons in candidate_neurons.items():
            c = centroids[layer_idx].reshape(1, -1) # centroid of the layer.
            # i-th layer hidden_state.
            ht_pre = ht_all_layer[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            # i-th layer attention output.
            atts = post_attention_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            # i-th layer activation values (act_fn(x) * up_proj(x))
            act_fn = MLP_act_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            up_proj = up_proj_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            acts = calc_element_wise_product(act_fn, up_proj)

            # layer_score.
            layer_score = (ht_pre + atts).reshape(1, -1) # H^l-1 * Att^l
            if score_type == "L2_dis":
                layer_score = euclidean_distances(layer_score, c)[0, 0]
            elif score_type == "cos_sim":
                layer_score = cosine_similarity(layer_score, c)[0, 0]

            neuron_indices = np.array(candidate_neurons[layer_idx])
            value_vectors = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[neuron_indices].detach().cpu().numpy()
            scores = (ht_pre + atts + (acts.reshape(-1, 1) * value_vectors))
            if score_type == "L2_dis":
                scores = euclidean_distances(scores, c).reshape(-1)
                scores = np.where(scores <= layer_score, abs(layer_score - scores), -abs(layer_score - scores))
            elif score_type == "cos_sim":
                scores = cosine_similarity(scores, c).reshape(-1)
                scores = np.where(scores >= layer_score, abs(layer_score - scores), -abs(layer_score - scores))
            # save resulting scores.
            final_scores_save[layer_idx, :, text_idx] = scores
            # final_scores_save[layer_idx, :, text_idx] = scores

    # Calculate mean score for each neuron
    final_scores[:, :] = np.mean(final_scores_save, axis=2)

    return final_scores

def compute_scores_optimized_bloom(model, tokenizer, device, data, candidate_neurons, centroids, score_type):
    """
    Optimized function to compute scores using batched inference.
    """
    num_layers = model.config.num_hidden_layers
    num_candidate_layers = len(candidate_neurons.keys())
    num_neurons = 10240
    final_scores_save = np.zeros((num_candidate_layers, num_neurons, len(data))) # final_scoreを計算する前の保存用
    final_scores = np.zeros((num_candidate_layers, num_neurons))
    num_dim_hs = 2560

    for text_idx, text in enumerate(data):
        # Tokenize all input data at once
        inputs = tokenizer(text, return_tensors="pt").to(device)
        # Extract hidden states (ht), attention layer outputs, and MLP activations
        token_len = inputs.input_ids.size(1)
        last_token_idx = token_len - 1

        # Score calculation based on type (L2 distance or cosine similarity)
        attn_output_array = np.zeros((num_candidate_layers, num_dim_hs))
        mlp_output_array = np.zeros((num_candidate_layers, num_neurons))
        handles = []
        for layer_idx, layer in enumerate(model.transformer.h):
            handle_attn = layer.self_attention.dense.register_forward_hook(
                lambda model, input, output, layer_idx=layer_idx, output_array=attn_output_array: get_attn_output_hook_for_transfer(model, input, output, layer_idx, output_array)
            )
            handles.append(handle_attn)
            handle_mlp = layer.mlp.dense_4h_to_h.register_forward_pre_hook(
                lambda model, input, layer_idx=layer_idx, output_array=mlp_output_array: get_activations_hook_for_transfer(model, input, layer_idx, output_array)
            )
            handles.append(handle_mlp)
        
        # run inference.
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
        
        # remove handles.
        for handle in handles:
            handle.remove()

        ht_all_layer = output.hidden_states # len==41(0-40), including 0th layer(embedding layer). <- emb(0th) layer is needed to calc score of 1th layer.
        for layer_idx, neurons in candidate_neurons.items():
            c = centroids[layer_idx].reshape(1, -1) # centroid of the layer.
            # i-1th layer hidden_state.
            ht_pre = ht_all_layer[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            # i-th layer attention output.
            atts = attn_output_array[layer_idx, :]
            # i-th layer activation values (act_fn(x) * up_proj(x))
            acts = mlp_output_array[layer_idx, :]
            
            # layer_score.
            layer_score = (ht_pre + atts).reshape(1, -1) # H^l-1 * Att^l
            if score_type == "L2_dis":
                layer_score = euclidean_distances(layer_score, c)[0, 0]
            elif score_type == "cos_sim":
                layer_score = cosine_similarity(layer_score, c)[0, 0]

            neuron_indices = np.array(candidate_neurons[layer_idx])
            value_vectors = model.transformer.h[layer_idx].mlp.dense_4h_to_h.weight.T.data[neuron_indices].detach().cpu().numpy()
            scores = (ht_pre + atts + (acts.reshape(-1, 1) * value_vectors))
            if score_type == "L2_dis":
                scores = euclidean_distances(scores, c).reshape(-1)
                scores = np.where(scores <= layer_score, abs(layer_score - scores), -abs(layer_score - scores))
            elif score_type == "cos_sim":
                scores = cosine_similarity(scores, c).reshape(-1)
                scores = np.where(scores >= layer_score, abs(layer_score - scores), -abs(layer_score - scores))

            # save resulting scores.
            final_scores_save[layer_idx, :, text_idx] = scores

    # Calculate mean score for each neuron
    final_scores[:, :] = np.mean(final_scores_save, axis=2)

    return final_scores

def compute_scores_optimized_qwen(model, tokenizer, device, data, candidate_neurons, centroids, score_type):
    """
    Optimized function to compute scores using batched inference.
    """
    num_layers = model.config.num_hidden_layers
    num_candidate_layers = len(candidate_neurons.keys())
    num_neurons = 12288
    final_scores_save = np.zeros((num_candidate_layers, num_neurons, len(data))) # final_scoreを計算する前の保存用
    final_scores = np.zeros((num_candidate_layers, num_neurons))

    for text_idx, text in enumerate(data):
        # Tokenize all input data at once
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Perform the forward pass once to get all necessary values
        MLP_act_values, up_proj_values, post_attention_values, outputs = get_all_outputs_llama3_mistral(model, inputs.input_ids, device)

        # Extract hidden states (ht), attention layer outputs, and MLP activations
        ht_all_layer = outputs.hidden_states # len==33(0-32), including 0th layer(embedding layer). <- emb(0th) layer is needed to calc score of 1th layer.
        token_len = inputs.input_ids.size(1)
        last_token_idx = token_len - 1

        # Score calculation based on type (L2 distance or cosine similarity)
        # c = np.mean(centroids[5:21], axis=0).reshape(1, -1) # centroids: mean of c for 5-20layers.
        for layer_idx, neurons in candidate_neurons.items():
            c = centroids[layer_idx].reshape(1, -1) # centroid of the layer.
            # i-th layer hidden_state.
            ht_pre = ht_all_layer[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            # i-th layer attention output.
            atts = post_attention_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            # i-th layer activation values (act_fn(x) * up_proj(x))
            act_fn = MLP_act_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            up_proj = up_proj_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            acts = calc_element_wise_product(act_fn, up_proj)

            # layer_score.
            layer_score = (ht_pre + atts).reshape(1, -1) # H^l-1 * Att^l
            if score_type == "L2_dis":
                layer_score = euclidean_distances(layer_score, c)[0, 0]
            elif score_type == "cos_sim":
                layer_score = cosine_similarity(layer_score, c)[0, 0]

            neuron_indices = np.array(candidate_neurons[layer_idx])
            value_vectors = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[neuron_indices].detach().cpu().numpy()
            scores = (ht_pre + atts + (acts.reshape(-1, 1) * value_vectors))
            if score_type == "L2_dis":
                scores = euclidean_distances(scores, c).reshape(-1)
                scores = np.where(scores <= layer_score, abs(layer_score - scores), -abs(layer_score - scores))
            elif score_type == "cos_sim":
                scores = cosine_similarity(scores, c).reshape(-1)
                scores = np.where(scores >= layer_score, abs(layer_score - scores), -abs(layer_score - scores))
            # save resulting scores.
            final_scores_save[layer_idx, :, text_idx] = scores
            # final_scores_save[layer_idx, :, text_idx] = scores

    # Calculate mean score for each neuron
    final_scores[:, :] = np.mean(final_scores_save, axis=2)

    return final_scores

def get_attn_output_hook_for_transfer(model, input, output, layer_idx: int, output_array: np.array):
    attn_output = output[0][-1, :].detach().cpu().numpy()
    output_array[layer_idx, :] = attn_output
def get_activations_hook_for_transfer(model, input, layer_idx: int, activation_array: np.array):
    activation_values = input[0][0, -1, :].detach().cpu().numpy()
    activation_array[layer_idx, :] = activation_values

def compute_scores_optimized_phi4(model, tokenizer, device, data, candidate_neurons, centroids, score_type):
    """
    Optimized function to compute scores using batched inference.
    """
    num_layers = model.config.num_hidden_layers
    num_candidate_layers = len(candidate_neurons.keys())
    num_neurons = 17920
    final_scores_save = np.zeros((num_candidate_layers, num_neurons, len(data))) # final_scoreを計算する前の保存用
    final_scores = np.zeros((num_candidate_layers, num_neurons))
    num_dim_hs = 5120

    for text_idx, text in enumerate(data):
        # Tokenize all input data at once
        inputs = tokenizer(text, return_tensors="pt").to(device)
        # Extract hidden states (ht), attention layer outputs, and MLP activations
        token_len = inputs.input_ids.size(1)
        last_token_idx = token_len - 1

        # Score calculation based on type (L2 distance or cosine similarity)
        attn_output_array = np.zeros((num_candidate_layers, num_dim_hs))
        mlp_output_array = np.zeros((num_candidate_layers, num_neurons))
        handles = []
        for layer_idx, layer in enumerate(model.model.layers):
            handle_attn = layer.self_attn.o_proj.register_forward_hook(
                lambda model, input, output, layer_idx=layer_idx, output_array=attn_output_array: get_attn_output_hook_for_transfer(model, input, output, layer_idx, output_array)
            )
            handles.append(handle_attn)
            handle_mlp = layer.mlp.down_proj.register_forward_pre_hook(
                lambda model, input, layer_idx=layer_idx, output_array=mlp_output_array: get_activations_hook_for_transfer(model, input, layer_idx, output_array)
            )
            handles.append(handle_mlp)
        
        # run inference.
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
        
        # remove handles.
        for handle in handles:
            handle.remove()

        ht_all_layer = output.hidden_states # len==41(0-40), including 0th layer(embedding layer). <- emb(0th) layer is needed to calc score of 1th layer.
        for layer_idx, neurons in candidate_neurons.items():
            c = centroids[layer_idx].reshape(1, -1) # centroid of the layer.
            # i-1th layer hidden_state.
            ht_pre = ht_all_layer[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
            # i-th layer attention output.
            atts = attn_output_array[layer_idx, :]
            # i-th layer activation values (act_fn(x) * up_proj(x))
            acts = mlp_output_array[layer_idx, :]
            
            # layer_score.
            layer_score = (ht_pre + atts).reshape(1, -1) # H^l-1 * Att^l
            if score_type == "L2_dis":
                layer_score = euclidean_distances(layer_score, c)[0, 0]
            elif score_type == "cos_sim":
                layer_score = cosine_similarity(layer_score, c)[0, 0]

            neuron_indices = np.array(candidate_neurons[layer_idx])
            value_vectors = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[neuron_indices].detach().cpu().numpy()
            scores = (ht_pre + atts + (acts.reshape(-1, 1) * value_vectors))
            if score_type == "L2_dis":
                scores = euclidean_distances(scores, c).reshape(-1)
                scores = np.where(scores <= layer_score, abs(layer_score - scores), -abs(layer_score - scores))
            elif score_type == "cos_sim":
                scores = cosine_similarity(scores, c).reshape(-1)
                scores = np.where(scores >= layer_score, abs(layer_score - scores), -abs(layer_score - scores))

            # save resulting scores.
            final_scores_save[layer_idx, :, text_idx] = scores

    # Calculate mean score for each neuron
    final_scores[:, :] = np.mean(final_scores_save, axis=2)

    return final_scores

# def compute_scores_optimized(model, tokenizer, device, data, candidate_neurons, centroids, score_type):
#     """
#     Optimized function to compute scores using batched inference.
#     """
#     num_layers = model.config.num_hidden_layers
#     num_candidate_layers = len(candidate_neurons.keys())
#     num_neurons = 14336
#     final_scores_save = np.zeros((num_candidate_layers, num_neurons, len(data))) # final_scoreを計算する前の保存用
#     final_scores = np.zeros((num_candidate_layers, num_neurons))

#     for text_idx, text in enumerate(data):
#         # Tokenize all input data at once
#         inputs = tokenizer(text, return_tensors="pt").to(device)

#         # Perform the forward pass once to get all necessary values
#         MLP_act_values, up_proj_values, post_attention_values, outputs = get_all_outputs_llama3_mistral(model, inputs.input_ids, device)

#         # Extract hidden states (ht), attention layer outputs, and MLP activations
#         ht_all_layer = outputs.hidden_states[1:] # remove 0th layer(embedding layer).
#         token_len = inputs.input_ids.size(1)
#         last_token_idx = token_len - 1

#         # Score calculation based on type (L2 distance or cosine similarity)
#         # c = np.mean(centroids[5:21], axis=0).reshape(1, -1) # centroids: mean of c for 5-20layers.
#         for layer_idx, neurons in candidate_neurons.items():
#             c = centroids[layer_idx].reshape(1, -1) # centroid of the layer.
#             # i-th layer hidden_state.
#             ht_pre = ht_all_layer[layer_idx-1][:, last_token_idx, :].squeeze().detach().cpu().numpy()
#             # i-th layer attention output.
#             atts = post_attention_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
#             # i-th layer activation values (act_fn(x) * up_proj(x))
#             act_fn = MLP_act_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
#             up_proj = up_proj_values[layer_idx][:, last_token_idx, :].squeeze().detach().cpu().numpy()
#             acts = calc_element_wise_product(act_fn, up_proj)

#             # layer_score.
#             layer_score = (ht_pre + atts).reshape(1, -1) # H^l-1 * Att^l
#             if score_type == "L2_dis":
#                 layer_score = euclidean_distances(layer_score, c)[0, 0]
#             elif score_type == "cos_sim":
#                 layer_score = cosine_similarity(layer_score, c)[0, 0]

#             # score for each neurons.
#             for neuron_idx in neurons:
#                 # get value vector cerresponding to the neuron.
#                 value_vector = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[neuron_idx].detach().cpu().numpy()
#                 score = (ht_pre + atts + (acts[neuron_idx] * value_vector)).reshape(1, -1)
                
#                 if score_type == "L2_dis":
#                     score = euclidean_distances(score, c)[0, 0]
#                     score = abs(layer_score - score) if score <= layer_score else abs(layer_score - score) * -1
#                 elif score_type == "cos_sim":
#                     score = cosine_similarity(score, c)[0, 0]
#                     score = abs(layer_score - score) if score >= layer_score else abs(layer_score - score) * -1
#                 final_scores_save[layer_idx, neuron_idx, text_idx] = score

#     # Calculate mean score for each neuron
#     final_scores[:, :] = np.mean(final_scores_save, axis=2)
#     # for layer_idx in range(num_candidate_layers):
#     #     for neuron_idx in range(num_neurons):
#     #         mean_score = np.mean(final_scores_save[layer_idx, neuron_idx, :])
#     #         final_scores[layer_idx, neuron_idx, 0] = mean_score

#     return final_scores

def sort_neurons_by_score(final_scores):
    # # (layer_num, neuron_num, 1) → (layer_num, neuron_num) に変換
    # final_scores_2d = final_scores.squeeze(-1)

    # {(layer_idx, neuron_idx): score} の辞書作成
    score_dict = {
        (layer_idx, neuron_idx): final_scores[layer_idx, neuron_idx]
        for layer_idx in range(final_scores.shape[0])
        for neuron_idx in range(final_scores.shape[1])
    }
    # スコアの降順ソート: [(layer_idx, neuron_idx), ...]
    sorted_neurons = sorted(score_dict.keys(), key=lambda x: score_dict[x], reverse=True)

    return sorted_neurons, score_dict

def get_out_llama3_post_attention_layernorm(model, prompt, device, index):
    model.eval() # swith the model to evaluation mode (deactivate dropout, batch normalization)
    num_layers = model.config.num_hidden_layers  # nums of layers of the model
    ATT_act = [f"model.layers.{i}.self_attn.o_proj" for i in range(num_layers)]  # generate path to MLP layer(of LLaMA-3)

    torch.manual_seed(42)
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

def compute_scores(model, tokenizer, device, data, candidate_neurons, centroids, score_type):
    """
    data: texts in specific L2 ([txt1, txt2, ...]).
    candidate_neurons: [(l, i), (l, i), ...] l: layer_idx, i: neuron_idx.
    """
    num_layers = model.config.num_hidden_layers
    final_scores = {} # { (l, i): [score1, score2, ...] }

    torch.manual_seed(42)
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

        if score_type == "L2_dis":
            # calc scores: L2_dist((hs^l-1 + self-att() + a^l_i*v^l_i) - c) <- layerごとにあまり差が出ないので、layerの平均からの差を見る.
            c = np.mean(centroids[5:21], axis=0).reshape(1, -1)
            for layer_idx, neurons in candidate_neurons.items():
                layer_score = (hts[layer_idx-1] + atts[layer_idx]).reshape(1, -1)
                layer_score = euclidean_distances(layer_score, c)[0, 0]
                for neuron_idx in neurons:
                    value_vector = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[neuron_idx].cpu().numpy()
                    score = (hts[layer_idx-1] + atts[layer_idx] + (acts[layer_idx][neuron_idx]*value_vector)).reshape(1, -1)
                    score = euclidean_distances(score, c)[0, 0]
                    score = abs(layer_score - score) if score <= layer_score else abs(layer_score - score)*-1
                    final_scores.setdefault((layer_idx, neuron_idx), []).append(score)
        elif score_type == "cos_sim":
            # calc scores: cos_sim((hs^l-1 + self-att() + a^l_i*v^l_i) - c) <- layerごとにあまり差が出ないので、layerの平均からの差を見る.
            c = np.mean(centroids[5:21], axis=0).reshape(1, -1)
            for layer_idx, neurons in candidate_neurons.items():
                layer_score = (hts[layer_idx-1] + atts[layer_idx]).reshape(1, -1)
                layer_score = cosine_similarity(layer_score, c)[0, 0]
                for neuron_idx in neurons:
                    value_vector = model.model.layers[layer_idx].mlp.down_proj.weight.T.data[neuron_idx].cpu().numpy()
                    score = (hts[layer_idx-1] + atts[layer_idx] + (acts[layer_idx][neuron_idx]*value_vector)).reshape(1, -1)
                    # score = value_vector.reshape(1, -1)
                    score = cosine_similarity(score, c)[0, 0]
                    score = abs(layer_score - score) if score >= layer_score else abs(layer_score - score)*-1
                    final_scores.setdefault((layer_idx, neuron_idx), []).append(score)         

    # 
    for layer_neuron_idx, scores in final_scores.items():
        mean_score = np.mean(scores)
        final_scores[layer_neuron_idx] = mean_score

    return final_scores  

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

def compute_ap_and_sort_np(activations_array: np.array, labels: list):
    """
    calc APscore and sort (considering nums_of_label) for detecting L2-specific neurons.

    Args:
        activations: activation_value for each sentences.
        start_label1: indicating idx where label1(corresponding to L2) sentences bigin.

    Returns:
        sorted_neurons list, ニューロンごとのAPスコア辞書
    """
    num_layers = 32
    num_neurons = 14336

    # calc AP score for each shared neuron and calc total score which consider nums of label
    final_scores = {}
    for layer_idx in range(num_layers):
        for neuron_idx in range(num_neurons):
            activations = activations_array[layer_idx, neuron_idx, :]
            # calc AP score
            ap = average_precision_score(y_true=labels, y_score=activations)
            # save total score
            final_scores[(layer_idx, neuron_idx)] = ap

    # sort: based on total score of each neuron
    sorted_neurons = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)

    return sorted_neurons, final_scores

# visualization
def plot_hist(dict1, dict2, L2: str, AUC_or_AUC_baseline:str, intervention_num: str) -> None:
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
        f"/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/sim/llama3/{L2}_n{intervention_num}.png",
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
            output[:, -1, neuron_idx] *= 0

    return output

def take_similarities_with_edit_activation(model, model_type, tokenizer, device, layer_neuron_list, data):
    if model_type in ['llama3', 'mistral', 'aya', 'qwen']:
        trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]))
    elif model_type in ['phi4']:
        trace_layers = list(set([f'model.layers.{layer}.mlp.activation_fn' for layer, _ in layer_neuron_list]))
    similarities = defaultdict(list)

    torch.manual_seed(42)
    for L1_txt, L2_txt in data:
        # get L2 hs (with intervention).
        with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)):
            inputs_L2 = tokenizer(L2_txt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_L2 = model(**inputs_L2, output_hidden_states=True)
            last_token_index_L2 = inputs_L2["input_ids"].shape[1] - 1
            last_token_hidden_states_L2 = [
                layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy()
                for layer_hidden_state in output_L2.hidden_states
            ]

        # get L1 hs (without intervention).
        inputs_L1 = tokenizer(L1_txt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_L1 = model(**inputs_L1, output_hidden_states=True)
        last_token_index_L1 = inputs_L1["input_ids"].shape[1] - 1
        last_token_hidden_states_L1 = [
            layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy()
            for layer_hidden_state in output_L1.hidden_states
        ]

        # cos_sim for this sentence pair
        similarities = calc_cosine_sim(last_token_hidden_states_L1, last_token_hidden_states_L2, similarities)

    return similarities


# def take_similarities_with_edit_activation(model, tokenizer, device, layer_neuron_list, data):
#     trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]))
#     with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:

#         return calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, device, data)

def calc_similarities_of_hidden_state_per_each_sentence_pair(model, tokenizer, device, data):
    """
    各層について、2000文ペアそれぞれのhidden_statesの類似度の平均を計算
    """
    similarities = defaultdict(list) # {layer_idx: mean_sim_of_each_sentences}

    torch.manual_seed(42)
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
            layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy()
            for layer_hidden_state in all_hidden_states_L1
        ]
        last_token_hidden_states_L2 = [
            layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy()
            for layer_hidden_state in all_hidden_states_L2
        ]
        # last_token_hidden_states_L1 = [
        #     (layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy() /
        #     np.linalg.norm(layer_hidden_state[:, last_token_index_L1, :].detach().cpu().numpy(), axis=-1, keepdims=True))
        #     for layer_hidden_state in all_hidden_states_L1
        # ]
        # last_token_hidden_states_L2 = [
        #     (layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy() /
        #     np.linalg.norm(layer_hidden_state[:, last_token_index_L2, :].detach().cpu().numpy(), axis=-1, keepdims=True))
        #     for layer_hidden_state in all_hidden_states_L2
        # ]

        # cos_sim
        similarities = calc_cosine_sim(last_token_hidden_states_L1, last_token_hidden_states_L2, similarities)

    return similarities

def calc_cosine_sim(last_token_hidden_states_L1: list, last_token_hidden_states_L2: list, similarities):
    """
    calc similarity per layer.
    """
    for layer_idx, (hidden_state_L1, hidden_state_L2) in enumerate(zip(last_token_hidden_states_L1, last_token_hidden_states_L2)):
        sim = cosine_similarity(hidden_state_L1, hidden_state_L2)[0, 0] # <- [[0.50695133]] のようになっているので、数値の部分だけ抽出
        similarities[layer_idx].append(sim)

    return similarities

def save_np_arrays(save_path, np_array):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Save directly to .npz
        np.savez(save_path, data=np_array)
        print(f"Array successfully saved to {save_path}")
    except Exception as e:
        print(f"Failed to save array: {e}")

def unfreeze_np_arrays(save_path):
    try:
        with np.load(save_path) as data:
            return data["data"]
    except Exception as e:
        print(f"Failed to load array: {e}")
        return None

def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d