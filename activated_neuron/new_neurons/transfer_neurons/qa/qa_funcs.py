import os
import re
import random
import string
import sys
import collections
import pickle
from collections import Counter, defaultdict
from functools import partial

import cld3
import torch
import numpy as np
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict


""" Metrics for generated Answers. """

MIXED_SEGMENTATION_LANGS = ["zh_cn", "zh_hk", "zh_tw", "ja", "th", "km", "ko"]
ARTICLE_REGEX_BY_LANG = {
    "en": r"\b(a|an|the)\b",
    "es": r"\b(un|una|unos|unas|el|la|los|las)\b",
    "vi": r"\b(của|là|cái|chiếc|những)\b",
    "de": r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b",
    "ar": "\sال^|ال",
    "nl": r"\b(de|het|een|des|der|den)\b",
    "sv": r"\b(en|ett)\b",
    "da": r"\b(en|et)\b",
    "no": r"\b(en|et|ei)\b",
    "fr": r"\b(le|la|l'|les|du|de|d'|des|un|une|des)",
    "pt": r"\b(o|a|os|as|um|uma|uns|umas)\b",
    "it": r"\b(il|lo|la|l'|i|gli|le|del|dello|della|dell'|dei|degli|degl'|delle|un'|uno|una|un)",
    "fi": r"\b(se|yks|yksi)\b",
    "hu": r"\b(a|az|egy)\b",
    "ko": r""  # 韓国語は冠詞がないので空にする,
}

def whitespace_tokenize(text):
    return text.split()

def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if temp_str != "":
            ss = whitespace_tokenize(temp_str)
            segs_out.extend(ss)
            temp_str = ""
        segs_out.append(char)

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out

def normalize_answer_by_language(s, lang):
    """Lower text, remove punctuation, articles and extra whitespace.
    This function is customized by language.
    """

    def remove_articles(text, lang):
        article_regex = ARTICLE_REGEX_BY_LANG.get(lang)
        if article_regex:
            return re.sub(article_regex, " ", text)
        else:
            return text

    def white_space_fix(text, lang):

        if lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            tokens = whitespace_tokenize(text)
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)

def calculate_f1(prediction, gold_answer, language):
    gold_toks = normalize_answer_by_language(gold_answer, language).split() if gold_answer else []
    pred_toks = normalize_answer_by_language(prediction, language).split() if prediction else []
    common = Counter(gold_toks) & Counter(pred_toks)
    num_common = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If the prediction or gold_answer is No Answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_common == 0:
        return 0.0

    recall = 1.0 * num_common / len(gold_toks)
    precision = 1.0 * num_common / len(pred_toks)
    return (2.0 * precision * recall) / (precision + recall)

# def compute_f1(a_gold, a_pred):
#     # gold_toks = get_tokens(a_gold)
#     # pred_toks = get_tokens(a_pred)
#     gold_toks = a_gold
#     pred_toks = a_pred
#     common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
#     num_same = sum(common.values())
#     if len(gold_toks) == 0 or len(pred_toks) == 0:
#         # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
#         return int(gold_toks == pred_toks)
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(pred_toks)
#     recall = 1.0 * num_same / len(gold_toks)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

""" """

""" func for editing activation values """
def edit_activation(output, layer, layer_idx_and_neuron_idx):
    """
    edit activation value of neurons(indexed layer_idx and neuron_idx)
    output: activation values
    layer: sth like 'model.layers.{layer_idx}.mlp.act_fn'
    layer_idx_and_neuron_idx: list of tuples like [(layer_idx, neuron_idx), ....]
    """
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if str(layer_idx) in layer and output.shape[1] != 1:
            output[:, -1, neuron_idx] *= 0

    return output

def mkqa_with_edit_activation(model, tokenizer, device, qa, L2, qa_num, layer_neuron_list):
    trace_layers = list(set([f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]))
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:

        return mkqa(model, tokenizer, device, qa, L2, qa_num)

def mkqa(model, tokenizer, device, qa, L2: str, qa_num: int):
    c = 0 # question counter.
    f1_scores = []
    for i in range(len(qa['queries'])):
        if c == qa_num: break
        q = qa['queries'][i][L2] # question
        if qa['answers'][i][L2][0]['aliases'] == []:
            a = [qa['answers'][i][L2][0]['text']] # answer as list.
        else:
            a = qa['answers'][i][L2][0]['aliases'] # answer as aliases: see: https://github.com/apple/ml-mkqa/tree/main?tab=readme-ov-file

        def contains_none_or_empty(lst):
            return any(x is None or x == '' for x in lst)
        if q == '' or q == None or contains_none_or_empty(a):
            continue

        # make prompt.
        if L2 == 'ja': prompt = f'{q}? 答え: '
        elif L2 == 'nl': prompt = f'{q}? Antwoord: '
        elif L2 == 'ko': prompt = f'{q}? 답변: '
        elif L2 == 'it': prompt = f'{q}? Risposta: '
        elif L2  == 'en': prompt = f'{q}? Answer: '

        # run inference.
        torch.cuda.manual_seed_all(42) # set seed.
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
        pre = tokenizer.decode(output[0], skip_special_tokens=True)
        # 
        if L2 == 'ja': pre = pre.split("答え: ")[-1].strip()
        if L2 == 'nl': pre = pre.split('Antwoord: ')[-1].strip()
        if L2 == 'ko': pre = pre.split('답변: ')[-1].strip()
        if L2 == 'it': pre = pre.split('Risposta: ')[-1].strip()
        if L2 == 'en': pre = pre.split('Answer: ')[-1].strip()
        
        if len(a) == 1:
            f1 = calculate_f1(a[0], pre, L2)
        else:
            f1_l = []
            for ans in a:
                f1_l.append(calculate_f1(ans, pre, L2))
            f1 = max(f1_l)

        f1_scores.append(f1)
        c += 1
        # print(f'question: {prompt}')
        # print(f'model ans: {pre}')
        # print(f'gorund truth: {a}')
        # print(f'f1: {f1}')
        # sys.exit()
    
    return np.mean(np.array(f1_scores))

""" steer output lang. """
def get_mean_act_value(neurons: list, lang: str, model_type: str):
    save_path_activations = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/activations/{lang}_last_token_elem_wise.npz"
    act_values_arr = unfreeze_np_arrays(save_path_activations)
    act_values_arr = np.mean(act_values_arr, axis=2)
    # act_values = {}
    # for layer_i, neuron_i in neurons:
    #     act_values[(layer_i, neuron_i)] = np.mean(act_values_arr[layer_i, neuron_i, :])
    
    return act_values_arr

def remove_intersec(list_a, list_b):
    set_b = set(list_b)  # リストBを集合に変換（検索を高速化）
    return [item for item in list_a if item not in set_b]  # Bにない要素を残す

def edit_activation_revised(output, layer, layer_idx_and_neuron_idx, last_token_idx, device, act_values):
    for act_mode, layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if str(layer_idx) in layer:  # layer名にlayer_idxが含まれているか確認
            if act_mode == 'de' and output.shape[1] == last_token_idx+1:
            # if act_mode == 'de':
                output[:, -1, neuron_idx] *= 0
            elif act_mode == 'ac' and output.shape[1] == last_token_idx+1:
            # elif act_mode == 'ac':
                output[:, -1, neuron_idx] *= 3
                # output[:, -1, neuron_idx] = torch.tensor(float(act_values[(layer_idx, neuron_idx)]), dtype=float)

    return output

def mkqa_for_steer_output_lang(model, tokenizer, device, qa, lang_deact: str, qa_num: int, neurons_zero=None, neurons_up=None, act_values=None):
    c = 0 # question counter.
    f1_scores = []
    for i in range(len(qa['queries'])):
        if c == qa_num: break
        q = qa['queries'][i+100][lang_deact] # question
        a = qa['answers'][i+100][lang_deact][0]['text'] # answer
        if q == '' or q == None or  a == '' or a == None:
            continue

        # make prompt.
        if lang_deact == 'ja': prompt= f'{q}? 答え: '
        elif lang_deact == 'nl': prompt= f'{q}? Antwoord: '
        elif lang_deact == 'ko': prompt= f'{q}? 답변: '
        elif lang_deact == 'it': prompt= f'{q}? Risposta: '
        prompt = f'Wat is de hoofdstad van Japan? Antwoord: '
        prompt = f'Wat is de hoofdstad van China? Antwoord: '
        prompt = f'Wat is de hoofdstad van Korea? Antwoord: '
        prompt = 'Wat is de hoofdstad van Italië? Antwoord: '
        prompt = 'Wat eet een Nederlander graag? Antwoord: '
        prompt = 'Welke taal spreekt men in België? Antwoord: '
        prompt = "Wat zijn enkele populaire toeristische attracties in New York City? Antwoord: "
        prompt = 'オランダの首都はどこですか 答え: '
        prompt = 'Qual è la capitale del Giappone? Risposta: '
        # prompt_tran
        # run inference.
        torch.cuda.manual_seed_all(42)
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        token_len = inputs.input_ids.size(1)
        last_token_idx = token_len - 1

        # run inference with steering activations.
        trace_layers_zero = list(set([f'model.layers.{layer}.mlp.act_fn' for _, layer, _ in neurons_zero]))
        trace_layers_up = list(set([f'model.layers.{layer}.mlp.act_fn' for _, layer, _ in neurons_up]))
        trace_layers = list(set(trace_layers_zero+trace_layers_up))

        with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation_revised(output, layer, neurons_zero+neurons_up, last_token_idx, device, act_values)) as tr:
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)

        pre = tokenizer.decode(output[0], skip_special_tokens=True) # model's prediction

        if lang_deact == 'ja': pre = pre.split("答え: ")[-1].strip()
        if lang_deact == 'nl': pre = pre.split('Antwoord: ')[-1].strip()
        if lang_deact == 'ko': pre = pre.split('답변: ')[-1].strip()
        if lang_deact == 'it': pre = pre.split('Risposta: ')[-1].strip()
        c += 1
        # print(f'question: {prompt}')
        # print(f'model ans: {pre}')
        # print(f'gorund truth: {a}')
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
    """ """
    lang_count = 0
    total_num = 0
    """ """
    c = 0 # question counter.
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
        c += 1
        
        print(f'question: {prompt}')
        print(f'model ans: {pre}')
        print(f'gorund truth: {a}')
        """ calc ratio of lang_activation in the model's output. """
        pred_lang = cld3.get_language(pre)
        if pred_lang.is_reliable:
            total_num += 1
            if pred_lang.language == L2:
                lang_count += 1
    
    return lang_count / total_num

def save_np_arrays(save_path, np_array):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Save directly to .npz
        np.savez(save_path, data=np_array)
        print(f"Array successfully saved to {save_path}")
    except Exception as e:
        print(f"Failed to save array: {e}")

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


""" act_values as mlp_activation against translation sentences. """

def get_all_outputs_llama3_mistral(model, prompt, device):
    num_layers = model.config.num_hidden_layers
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(num_layers)]
    with TraceDict(model, MLP_act) as ret:
        with torch.no_grad():
            outputs = model(**prompt, output_hidden_states=True, output_attentions=True)
    MLP_act_values = [ret[act].output for act in MLP_act]
    
    return MLP_act_values

def edit_activation_revised_act_value(output, layer, layer_idx_and_neuron_idx, device, act_values, last_token_idx):
    for act_mode, layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if str(layer_idx) in layer:  # layer名にlayer_idxが含まれているか確認
            if act_mode == 'de' and output.shape[1] == last_token_idx+1:
            # if act_mode == 'de':
                # output[:, -1, neuron_idx] *= 0
                output[:, -1, neuron_idx] = act_values[layer_idx][:, -1, neuron_idx]
            elif act_mode == 'ac' and output.shape[1] == last_token_idx+1:
            # elif act_mode == 'ac':
                output[:, -1, neuron_idx] = act_values[layer_idx][:, -1, neuron_idx]

    return output

def mkqa_for_steer_output_lang_act_values(model, tokenizer, device, qa, lang_deact: str, lang_act: str, qa_num: int, neurons_zero=None, neurons_up=None):
    c = 0 # question counter.
    f1_scores = []
    ans_patterns = {
    'ja': '答え: ',
    'nl': 'Antwoord: ',
    'ko': '답변: ',
    'it': 'Risposta: ',
    }
    for i in range(len(qa['queries'])):
        if c == qa_num: break
        q = qa['queries'][i][lang_deact] # question
        a = qa['answers'][i][lang_deact][0]['text'] # answer
        q_tran = qa['queries'][i][lang_act]
        a_tran = qa['answers'][i][lang_act][0]['text']
        if q == '' or q == None or  a == '' or a == None or q_tran == '' or q_tran == '' or q_tran == None or a_tran == None:
            continue

        # make prompt.
        prompt_lang_deact = f'{q}? {ans_patterns[lang_deact]}'
        prompt_lang_act = f'{q_tran}? {ans_patterns[lang_act]}'
        # prompt = f'Wat is de hoofdstad van Japan? Antwoord: '
        # prompt = f'Wat is de hoofdstad van China? Antwoord: '
        # prompt = f'Wat is de hoofdstad van Korea? Antwoord: '
        # prompt_lang_deact = 'Wat is de hoofdstad van Italië? Risposta: '
        # prompt = 'Wat eet een Nederlander graag? Antwoord: '
        # prompt_lang_deact = 'Welke taal spreekt men in België? Antwoord: '
        # prompt_lang_deact = "Wat zijn enkele populaire toeristische attracties in New York City? Antwoord: "
        # prompt_lang_act = 'イタリアの首都はどこですか 答え: '
        # prompt_lang_act = 'ベルギーでは何の言語が話されていますか? 答え: '
        # prompt = 'こんにちは、今日は'
        # prompt = 'Qual è la capitale del Giappone? Risposta: '
        # prompt_tran

        # run inference.
        torch.cuda.manual_seed_all(42)
        # lang_deact.
        inputs_lang_deact = tokenizer(prompt_lang_deact, return_tensors='pt').to(device)
        last_token_idx = inputs_lang_deact['input_ids'].shape[1] - 1
        # lang_act.
        inputs_lang_act = tokenizer(prompt_lang_act, return_tensors='pt').to(device)

        # run inference with steering activations.
        trace_layers_zero = list(set([f'model.layers.{layer}.mlp.act_fn' for _, layer, _ in neurons_zero]))
        trace_layers_up = list(set([f'model.layers.{layer}.mlp.act_fn' for _, layer, _ in neurons_up]))
        trace_layers = list(set(trace_layers_zero+trace_layers_up))

        # get act_values for lang_act.
        trace_layers_lang_act = [f'model.layers.{layer}.mlp.act_fn' for layer in range(32)]
        act_values = get_all_outputs_llama3_mistral(model, inputs_lang_act, device)

        with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation_revised_act_value(output, layer, neurons_zero+neurons_up, device, act_values, last_token_idx)) as tr:
            with torch.no_grad():
                output = model.generate(**inputs_lang_deact, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)

        pre = tokenizer.decode(output[0], skip_special_tokens=True) # model's prediction
        if lang_deact == 'ja': pre = pre.split("答え: ")[-1].strip()
        if lang_deact == 'nl': pre = pre.split('Antwoord: ')[-1].strip()
        if lang_deact == 'ko': pre = pre.split('답변: ')[-1].strip()
        if lang_deact == 'it': pre = pre.split('Risposta: ')[-1].strip()
        c += 1
        print(f'question: {prompt_lang_deact}')
        print(f'model ans: {pre}')
        print(f'gorund truth: {a}')
        # sys.exit()
    
    return np.mean(np.array(f1_scores))

""" add (c^l_lang2 - c^l_lang1) to hs of certain layer. """
def edit_activation_times(output, layer, layer_idx_and_neuron_idx, last_token_idx, device, act_values_act):
    for act_mode, layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if str(layer_idx) in layer:  # layer名にlayer_idxが含まれているか確認
            if act_mode == 'de' and output.shape[1] == last_token_idx+1:
                # output[:, -1, neuron_idx] *= 0
                # output[:, -1, neuron_idx] = torch.tensor(float(act_values_act[(layer_idx, neuron_idx)]), dtype=float)
                output[:, -1, neuron_idx] = act_values_act[layer_idx][neuron_idx]
                # output[:, -1, neuron_idx] = act_values_act[layer_idx][:, -1, neuron_idx]
            elif act_mode == 'ac' and output.shape[1] == last_token_idx+1:
                # output[:, -1, neuron_idx] *= 2
                # output[:, -1, neuron_idx] = torch.tensor(float(act_values_act[(layer_idx, neuron_idx)]), dtype=float)
                output[:, -1, neuron_idx] = act_values_act[layer_idx][neuron_idx]
                # output[:, -1, neuron_idx] = act_values_act[layer_idx][:, -1, neuron_idx]

    return output

def edit_activation_sub_vectors(output, layer, last_token_idx, device, sub_vectors: dict):
    for layer_idx, sub_vector in sub_vectors.items():
        if str(layer_idx) in layer and output[0].shape[1] == last_token_idx+1:  # layer名にlayer_idxが含まれているか確認
            output[0][:, -1, :] += torch.from_numpy(sub_vector).to(device)

    return output

def mkqa_for_steer_output_lang_add_subducted_vectors(
    model, tokenizer, device, qa, 
    lang_deact: str, lang_act, qa_num: int, 
    neurons_zero=None, neurons_up=None, 
    c_lang1=None, c_lang2=None,
    act_values_act=None,
    ):
    """ lang counter. """
    lang_count = 0
    total_num = 0
    # lang_deact = 'en'
    """ """
    c = 0 # question counter.
    for i in range(len(qa['queries'])):
        if c == qa_num: break
        q = qa['queries'][i+100][lang_deact] # question
        a = qa['answers'][i+100][lang_deact][0]['text'] # answer
        """ """
        q_tran = qa['queries'][i+100][lang_act]
        a_tran = qa['answers'][i+100][lang_act][0]['text']
        if q_tran == None or q_tran == '':
            continue
        """ """
        if q == '' or q == None or  a == '' or a == None:
            continue

        """ """
        ans_patterns = {
        'ja': '答え: ',
        'nl': 'Antwoord: ',
        'ko': '답변: ',
        'it': 'Risposta: ',
        'en': 'Answer: ',
        }
        prompt = f'{q}? {ans_patterns[lang_deact]}'
        prompt_lang_act = f'{q_tran}? {ans_patterns[lang_act]}'
        """ """
        # if lang_deact == 'ja': prompt= f'{q}? 答え: '
        # elif lang_deact == 'nl': prompt= f'{q}? Antwoord: '
        # elif lang_deact == 'ko': prompt= f'{q}? 답변: '
        # elif lang_deact == 'it': prompt= f'{q}? Risposta: '
        # prompt = f'Wat is de hoofdstad van Japan? Antwoord: '
        # prompt = f'Wat is de hoofdstad van China? Antwoord: '
        # prompt = f'Wat is de hoofdstad van Korea? Antwoord: '
        # prompt = 'Wat is de hoofdstad van Italië? Antwoord: '
        # prompt = 'Wat eet een Nederlander graag? Antwoord: '
        # prompt = 'Welke taal spreekt men in België? Antwoord: '
        # prompt = "Wat zijn enkele populaire toeristische attracties in New York City? Antwoord: "
        # prompt = 'オランダの首都はどこですか 答え: '
        # prompt = 'Qual è la capitale del Giappone? Risposta: '

        """ for tran. """
        inputs_lang_act = tokenizer(prompt_lang_act, return_tensors='pt').to(device)
        # trace_layers_lang_act = [f'model.layers.{layer}.mlp.act_fn' for layer in range(32)]
        # act_values = get_all_outputs_llama3_mistral(model, inputs_lang_act, device)
        
        torch.cuda.manual_seed_all(42)
        # # get elem_wise act_values.
        # act_values_act = [] # len: layer_num
        # # hook_fn for getting act_values.
        # def get_elem_wise_product(model, input):
        #     act_values_act.append(input[0][0][-1]) # last tokenに対応する活性値のみ取得
        # handles = []
        # for layer in model.model.layers:
        #     handle = layer.mlp.down_proj.register_forward_pre_hook(get_elem_wise_product)
        #     handles.append(handle)
        # # run inference.
        # with torch.no_grad():
        #     output = model(**inputs_lang_act)
        # # remove hook
        # for handle in handles:
        #     handle.remove()

        """ get subtracted vectors """
        sub_vectors = {}
        # target_layers = [ _ for _ in range(29, 32)] # Mistral
        # target_layers = [ _ for _ in range(30, 32)] # LLaMA3, Mistral
        # target_layers = [4, 15, 25, 31] # llama
        # target_layers = [9, 19, 24, 31] # 
        # target_layers = [19]
        # target_layers = [4, 9, 14, 19, 24, 29]
        """ lang_deactの方を実際にdeactivateしてみるといいかも """
        target_layers = []
        for target_layer in target_layers:
            sub_vector = c_lang2[target_layer] - c_lang1[target_layer]
            sub_vectors[target_layer] = sub_vector

        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        token_len = inputs.input_ids.size(1)
        last_token_idx = token_len - 1

        # run inference with steering activations.
        """ activation_values for activation patching. """
        # trace_layers_zero = list(set([f'model.layers.{layer}.mlp.act_fn' for _, layer, _ in neurons_zero]))
        # trace_layers_up = list(set([f'model.layers.{layer}.mlp.act_fn' for _, layer, _ in neurons_up]))
        # trace_layers = list(set(trace_layers_zero+trace_layers_up))
        # """ trace_layers for addition of subtracted_vectors """
        # trace_layers_add = [f'model.layers.{layer}' for layer in target_layers]

        """ for tran. """
        # with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation_times(output, layer, neurons_zero+neurons_up, last_token_idx, device, act_values_act)) as tr:
        # with TraceDict(model, trace_layers_add, edit_output=lambda output, layer: edit_activation_sub_vectors(output, layer, last_token_idx, device, sub_vectors)) as tr:
        #     with torch.no_grad():
        #         output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)

        neurons = defaultdict(list) # {layer_idx: [neuron_idx, ...]}
        for neuron in neurons_zero+neurons_up:
            neurons[neuron[1]].append(neuron[2])
        
        # hook_fn.
        def add_subtracted_vector(model, input, output, layer_idx: int, token_len: int):
            if layer_idx != 31:
                if output[0].shape[1] == token_len:
                    output[0][:, -1, :] += torch.from_numpy(sub_vectors[layer_idx]).to(device)

        def edit_elem_wise_product(model, input, layer_idx: int):
            if input[0].shape[1] == token_len:
                for neuron_idx in neurons[layer_idx]:
                    # for tran
                    # input[0][:, -1, neuron_idx] = act_values_act[layer_idx][neuron_idx] # last tokenに対応する活性値のみ取得
                    # for mean.
                    input[0][:, -1, neuron_idx] = torch.tensor(float(act_values_act[layer_idx, neuron_idx]), dtype=torch.float32, device=device)
        
        # register hook.
        handles = []
        for layer_idx, layer in enumerate(model.model.layers):
            # for patching act_value with translation_version.
            if layer_idx >= 20 and layer_idx <= 31:
                # handle = layer.mlp.down_proj.register_forward_pre_hook(
                #     partial(edit_elem_wise_product, layer_idx=layer_idx)
                # )
                handle = layer.mlp.down_proj.register_forward_pre_hook(
                    lambda model, input, layer_idx=layer_idx: edit_elem_wise_product(model, input, layer_idx)
                )
                handles.append(handle)
            # # for adding subtracted vector to the hidden_states.
            if layer_idx in target_layers:
                # handle2 = layer.register_forward_hook(
                #     partial(add_subtracted_vector, layer_idx=layer_idx, token_len=token_len)
                #     )
                handle2 = layer.register_forward_hook(
                    lambda model, input, output, layer_idx=layer_idx, token_len=token_len: add_subtracted_vector(model, input, output, layer_idx, token_len)
                )
                handles.append(handle2)

        # run inference.
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        # remove hook
        for handle in handles:
            handle.remove()

        pre = tokenizer.decode(output[0], skip_special_tokens=True) # model's prediction

        if lang_deact == 'ja': pre = pre.split("答え: ")[-1].strip()
        if lang_deact == 'nl': pre = pre.split('Antwoord: ')[-1].strip()
        if lang_deact == 'ko': pre = pre.split('답변: ')[-1].strip()
        if lang_deact == 'it': pre = pre.split('Risposta: ')[-1].strip()
        if lang_deact == 'en': pre = pre.split('Answer: ')[-1].strip()
        c += 1
        print(f'question: {prompt}')
        print(f'model ans: {pre}')
        print(f'gorund truth: {a}')

        """ calc ratio of lang_activation in the model's output. """
        pred_lang = cld3.get_language(pre)
        if pred_lang.is_reliable:
            total_num += 1
            if pred_lang.language == lang_act:
                lang_count += 1
        # print('\n', '====================================')
        # print(f'pred_lang: {pred_lang.language}')
        # print(f'lang_act: {lang_act}')
        # print(f'lang_count: {lang_count}')
        # print(f'total_num: {total_num}')
        # print(f'lang: {pred_lang}')
        
    
    return lang_count / total_num 

""" patching with elem_wise_product. """
def mkqa_for_steer_output_lang_pathing_with_elem_wise(
    model, tokenizer, device, qa, 
    lang_deact: str, lang_act, qa_num: int, 
    neurons_zero=None, neurons_up=None, 
    c_lang1=None, c_lang2=None,
    act_values_act=None,
    ):

    # hook fn.
    def edit_activations(model, input, layer_idx, neurons: list[int], token_len: int, act_array: np.array):
        input = input[0]
        if len(input[0]) == token_len:
            for neuron_idx in neurons:
                input[0][-1][neuron_idx] = torch.from_numpy(act_array[layer_idx, neuron_idx]).to(device)

    """ lang counter. """
    lang_count = 0
    total_num = 0
    """ """
    c = 0 # question counter.
    f1_scores = []
    for i in range(len(qa['queries'])):
        if c == qa_num: break
        q = qa['queries'][i+100][lang_deact] # question
        a = qa['answers'][i+100][lang_deact][0]['text'] # answer
        """ """
        q_tran = qa['queries'][i+100][lang_act]
        a_tran = qa['answers'][i+100][lang_act][0]['text']
        if q_tran == None or q_tran == '':
            continue
        """ """
        if q == '' or q == None or  a == '' or a == None:
            continue

        """ """
        ans_patterns = {
        'ja': '答え: ',
        'nl': 'Antwoord: ',
        'ko': '답변: ',
        'it': 'Risposta: ',
        }
        prompt = f'{q}? {ans_patterns[lang_deact]}'

        """ for tran. """
        # prompt_lang_act = f'{q_tran}? {ans_patterns[lang_act]}'
        # inputs_lang_act = tokenizer(prompt_lang_act, return_tensors='pt').to(device)
        # trace_layers_lang_act = [f'model.layers.{layer}.mlp.act_fn' for layer in range(32)]
        # act_values = get_all_outputs_llama3_mistral(model, inputs_lang_act, device)

        # run inference.
        torch.cuda.manual_seed_all(42)
        """ get subtracted vectors """
        sub_vectors = {}
        target_layers = [19]
        for target_layer in target_layers:
            sub_vector = c_lang2[target_layer] - c_lang1[target_layer]
            sub_vectors[target_layer] = sub_vector

        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        token_len = inputs.input_ids.size(1)
        last_token_idx = token_len - 1

        # run inference with steering activations.
        """ activation_values for activation patching. """
        trace_layers_zero = list(set([f'model.layers.{layer}.mlp.act_fn' for _, layer, _ in neurons_zero]))
        trace_layers_up = list(set([f'model.layers.{layer}.mlp.act_fn' for _, layer, _ in neurons_up]))
        trace_layers = list(set(trace_layers_zero+trace_layers_up))
        """ trace_layers for addition of subtracted_vectors """
        trace_layers_add = [f'model.layers.{layer}' for layer in target_layers]

        # with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation_times(output, layer, neurons_zero+neurons_up, last_token_idx, device, act_values_act)) as tr:
        """ for tran. """
        with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation_times(output, layer, neurons_zero+neurons_up, last_token_idx, device, act_values_act)) as tr:
            with TraceDict(model, trace_layers_add, edit_output=lambda output, layer: edit_activation_sub_vectors(output, layer, last_token_idx, device, sub_vectors)) as tr:
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)

        pre = tokenizer.decode(output[0], skip_special_tokens=True) # model's prediction

        if lang_deact == 'ja': pre = pre.split("答え: ")[-1].strip()
        if lang_deact == 'nl': pre = pre.split('Antwoord: ')[-1].strip()
        if lang_deact == 'ko': pre = pre.split('답변: ')[-1].strip()
        if lang_deact == 'it': pre = pre.split('Risposta: ')[-1].strip()
        # pre = pre.split('A: ')[-1].strip()
        c += 1
        print(f'question: {prompt}')
        print(f'model ans: {pre}')
        print(f'gorund truth: {a}')

        """ calc ratio of lang_activation in the model's output. """
        pred_lang = cld3.get_language(pre)
        if pred_lang.is_reliable:
            total_num += 1
            if pred_lang.language == lang_act:
                lang_count += 1
        # print('\n', '====================================')
        # print(f'pred_lang: {pred_lang.language}')
        # print(f'lang_act: {lang_act}')
        # print(f'lang_count: {lang_count}')
        # print(f'total_num: {total_num}')
        # print(f'lang: {pred_lang}')
        
    
    return lang_count / total_num