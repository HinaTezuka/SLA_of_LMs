"""
compare acc of QA for both normal and deactivated model.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import collections

import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

from qa_funcs import (
    compute_f1,
    generate_answer_ja,
    collect_qa_ans_ja,
    mkqa_ja,
)

# load models (LLaMA3-8B).
model_name = 'meta-llama/Meta-Llama-3-8B'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

""" 
QA dataset: 
ja: JaQuAD (https://github.com/SkelterLabsInc/JaQuAD / https://huggingface.co/datasets/SkelterLabsInc/JaQuAD)
"""
qa_num = 100

""" 
Metric: F1-score.
"""
# def collect_qa_ans_nl(model, tokenizer, device, qa_num):
#     qa = 
    


""" """
# Process each QA pair and generate answers
# score_ja = collect_qa_ans_ja(model, tokenizer, device, qa_num)
score_ja = mkqa_ja(model, tokenizer, device, qa_num)
print(score_ja)