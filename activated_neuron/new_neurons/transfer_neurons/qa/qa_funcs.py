import os
import re
import random
import sys
import collections
import pickle

import torch
import numpy as np
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict

def compute_f1(a_gold, a_pred):
    # gold_toks = get_tokens(a_gold)
    # pred_toks = get_tokens(a_pred)
    gold_toks = a_gold
    pred_toks = a_pred
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

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
            # output[:, :, neuron_idx] *= 0  # 指定されたニューロンの活性化値をゼロに設定
            output[:, -1, neuron_idx] *= 0

    return output

def mkqa_with_edit_activation(model, tokenizer, device, qa, L2, qa_num, layer_neuron_list):
    trace_layers = [f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:

        return mkqa(model, tokenizer, device, qa, L2, qa_num)

def mkqa(model, tokenizer, device, qa, L2: str, qa_num: int):
    c = 0 # question counter.
    f1_scores = []
    for i in range(len(qa['queries'])):
        if c == qa_num: break
        q = qa['queries'][i][L2] # question
        a = qa['answers'][i][L2][0]['text'] # answer
        if q == '' or q == None or  a == '' or a == None:
            continue

        # make prompt.
        if L2 == 'ja': prompt = f'{q} 答え: '
        elif L2 == 'nl': prompt = f'{q} Antwoord: '
        elif L2 == 'ko': prompt = f'{q} 답변: '
        elif L2 == 'it': prompt = f'{q} Risposta: '

        # run inference.
        torch.cuda.manual_seed_all(42) # set seed.
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        pre = tokenizer.decode(output[0], skip_special_tokens=True)
        # 
        if L2 == 'ja': pre = pre.split("答え: ")[-1].strip()
        if L2 == 'nl': pre = pre.split('Antwoord: ')[-1].strip()
        if L2 == 'ko': pre = pre.split('답변: ')[-1].strip()
        if L2 == 'it': pre = pre.split('Risposta: ')[-1].strip()
        f1 = compute_f1(a, pre)
        f1_scores.append(f1)
        c += 1
        # print(f'question: {q}')
        # print(f'model ans: {pre}')
        # print(f'gorund truth: {a}')
        # print(f'f1: {f1}')
    
    return np.mean(np.array(f1_scores))

""" steer output lang. """
def get_mean_act_value(neurons: list, lang: str, model_type: str):
    start_indics = {
        "ja": 0,
        "nl": 1000,
        "ko": 2000,
        "it": 3000,
        "en": 4000,
    }
    save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/activations/{lang}_last_token.npz"
    act_values_arr = unfreeze_np_arrays(save_path_activations)
    act_values = {}
    start_idx = start_indics[lang]
    end_idx = start_indics[lang] + 1000
    for layer_i, neuron_i in neurons:
        # print(len(act_values_arr[layer_i, neuron_i, :1000]))
        act_values[(layer_i, neuron_i)] = np.mean(act_values_arr[layer_i, neuron_i, start_idx:end_idx])
        # act_values[(layer_i, neuron_i)] = np.max(act_values_arr[layer_i, neuron_i, :1000])
    
    return act_values

def remove_intersec(list_a, list_b):
    set_b = set(list_b)  # リストBを集合に変換（検索を高速化）
    return [item for item in list_a if item not in set_b]  # Bにない要素を残す

def edit_activation_revised(output, layer, layer_idx_and_neuron_idx, last_token_idx, device):
    for act_mode, layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if str(layer_idx) in layer:  # layer名にlayer_idxが含まれているか確認
            if act_mode == 'de':
                output[:, -1, neuron_idx] *= 0.5
            elif act_mode == 'ac':
                output[:, -1, neuron_idx] *= 2

    return output

def mkqa_for_steer_output_lang(model, tokenizer, device, qa, lang_deact: str, qa_num: int, neurons_zero=None, neurons_up=None):
    c = 0 # question counter.
    f1_scores = []
    for i in range(len(qa['queries'])):
        if c == qa_num: break
        q = qa['queries'][i][lang_deact] # question
        a = qa['answers'][i][lang_deact][0]['text'] # answer
        if q == '' or q == None or  a == '' or a == None:
            continue

        # 対訳文を入れて、発火値を記録しておいて、書き換える？
        # make prompt.
        if lang_deact == 'ja': prompt= f'{q}? 答え: '
        elif lang_deact == 'nl': prompt= f'{q}? Antwoord: '
        elif lang_deact == 'ko': prompt= f'{q}? 답변: '
        elif lang_deact == 'it': prompt= f'{q}? Risposta: '
        # prompt = f'Wat is de hoofdstad van Japan? Antwoord: '
        prompt = f'Wat is de hoofdstad van China? Antwoord: '
        # prompt = f'Wat is de hoofdstad van Korea? Antwoord: '
        # prompt = 'Wat eet een Nederlander graag? Antwoord: '
        prompt = 'Welke taal spreekt men in België? Antwoord: '
        # prompt = "Wat zijn enkele populaire toeristische attracties in New York City? Antwoord: "
        # prompt = 'オランダの首都はどこですか 答え: '
        # prompt = 'こんにちは、今日は'
        # prompt = 'Qual è la capitale del Giappone? Risposta: '
        # prompt_tran
        # run inference.
        torch.cuda.manual_seed_all(42)
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        last_token_idx = inputs['input_ids'].shape[1] - 1

        # run inference with steering activations.
        trace_layers_zero = list(set([f'model.layers.{layer}.mlp.act_fn' for _, layer, _ in neurons_zero]))
        trace_layers_up = list(set([f'model.layers.{layer}.mlp.act_fn' for _, layer, _ in neurons_up]))
        trace_layers = list(set(trace_layers_zero+trace_layers_up))

        with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation_revised(output, layer, neurons_zero+neurons_up, last_token_idx, device)) as tr:
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)

        pre = tokenizer.decode(output[0], skip_special_tokens=True) # model's prediction
        if lang_deact == 'ja': pre = pre.split("答え: ")[-1].strip()
        if lang_deact == 'nl': pre = pre.split('Antwoord: ')[-1].strip()
        if lang_deact == 'ko': pre = pre.split('답변: ')[-1].strip()
        if lang_deact == 'it': pre = pre.split('Risposta: ')[-1].strip()
        c += 1
        print(f'question: {prompt}')
        print(f'model ans: {pre}')
        print(f'gorund truth: {a}')
        # sys.exit()
    
    return np.mean(np.array(f1_scores))

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

def mkqa_for_steer_output_lang_normal(model, tokenizer, device, qa, L2: str, qa_num: int):
    c = 0 # question counter.
    f1_scores = []
    for i in range(len(qa['queries'])):
        if c == qa_num: break
        q = qa['queries'][i][L2] # question
        a = qa['answers'][i][L2][0]['text'] # answer
        if q == '' or q == None or  a == '' or a == None:
            continue

        # make prompt.
        if L2 == 'ja': prompt = f'{q} 答え: '
        elif L2 == 'nl': prompt = f'{q} Antwoord: '
        elif L2 == 'ko': prompt = f'{q} 답변: '
        elif L2 == 'it': prompt = f'{q} Risposta: '
        # prompt = "Wat zijn enkele populaire toeristische attracties in New York City? Antwoord: "
        # prompt = f'Wat is de hoofdstad van Japan\nAntwoord: '
        # prompt = 'Wat is de hoofdstad van Nederland\nAntwoord: '
        # prompt = '日本の首都はどこですか 答え: '
        # prompt = 'Qual è la capitale del Giappone? Risposta: '

        # run inference.
        torch.cuda.manual_seed_all(42) # set seed.
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        last_token_idx = inputs['input_ids'].shape[1] - 1

        # with TraceDict(model, trace_layers_up, edit_output=lambda output, layer: edit_activation_up(output, layer, neurons_up, last_token_idx)) as tr:
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)

        pre = tokenizer.decode(output[0], skip_special_tokens=True)
        # 
        if L2 == 'ja': pre = pre.split("答え: ")[-1].strip()
        if L2 == 'nl': pre = pre.split('Antwoord: ')[-1].strip()
        if L2 == 'ko': pre = pre.split('답변: ')[-1].strip()
        if L2 == 'it': pre = pre.split('Risposta: ')[-1].strip()
        f1 = compute_f1(a, pre)
        f1_scores.append(f1)
        c += 1
        print(f'question: {prompt}')
        print(f'model ans: {pre}')
        print(f'gorund truth: {a}')
        # print(f'f1: {f1}')
        # sys.exit()
    
    return np.mean(np.array(f1_scores))

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

def unfreeze_np_arrays(save_path):
    try:
        with np.load(save_path) as data:
            return data["data"]
    except Exception as e:
        print(f"Failed to load array: {e}")
        return None