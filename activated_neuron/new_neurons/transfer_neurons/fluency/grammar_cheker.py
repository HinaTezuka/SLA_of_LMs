import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

import language_tool_python
from language_tool_python.utils import classify_matches
import cld3

from funcs import (
    save_as_pickle,
    unfreeze_pickle,
)

model_types = ['llama3', 'aya', 'mistral']
langs = ['fr', 'ja']
is_baselines = [False, True]
num_to_extract = 100

def lang_detection(sentence: str) -> bool: # 対象言語か/そもそも文字列が言語か　を判別用.
    return cld3.get_language(sentence)

for model_type in model_types:
    for L2 in langs:
        tool = language_tool_python.LanguageTool('fr-FR' if L2 == 'fr' else 'ja-JP')
        for is_baseline in is_baselines:
            save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{L2}.pkl' if not is_baseline else f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/mmluprox/{model_type}/{L2}_baseline.pkl'
            sentences = unfreeze_pickle(save_path)
            cnt = 0 # number of sentences judged gramatically correct.
            for sentence_i, sentence in enumerate(sentences):
                pred_language = lang_detection(sentence)
                if pred_language and pred_language.is_reliable and pred_language.language == L2:
                    try:
                        matches = tool.check(sentence)
                        if len(matches) == 0:
                            cnt += 1
                    except Exception as e:
                        print(f"Error on sentence {sentence_i}: {e}")
            """ print results. """
            print(f'------------ {model_type}, {L2}, {"Baseline" if is_baseline else "Type-1"} ------------')
            print(f'proportion: {cnt / num_to_extract}, {cnt}/{num_to_extract}')