import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import collections
import random
from collections import defaultdict

from datasets import load_dataset

from qa_funcs import (
    save_as_pickle,
    unfreeze_pickle,
)

c = 0 # question counter.
f1_scores = []
ans_patterns = {
'ja': '答え: ',
'nl': 'Antwoord: ',
'ko': '답변: ',
'it': 'Risposta: ',
}
langs = ['ja', 'nl', 'ko', 'it']
# load QA dataset.
qa_num = 4000 # train: 2000, test: 2000.
qa = load_dataset('apple/mkqa')['train']
qa = qa.shuffle(seed=42)

train = defaultdict(list) # {'ja': [(prompt, ans), (), ...], 'nl': [(prompt, ans)], ...}
test = defaultdict(list)
c = 1

for i in range(len(qa['queries'])):
    if c == qa_num: break
    q_ja = qa['queries'][i]['ja'] # question
    a_ja = qa['answers'][i]['ja'][0]['text'] # answer
    q_nl = qa['queries'][i]['nl']
    a_nl = qa['answers'][i]['nl'][0]['text']
    q_ko = qa['queries'][i]['ko']
    a_ko = qa['answers'][i]['ko'][0]['text']
    q_it = qa['queries'][i]['it']
    a_it = qa['answers'][i]['it'][0]['text']

    if any(not v for v in [q_ja, a_ja, q_nl, a_nl, q_ko, a_ko, q_it, a_it]):
        continue

    # make prompt.
    prompt_ja = f"{q_ja}? {ans_patterns['ja']}"
    prompt_nl = f"{q_nl}? {ans_patterns['nl']}"
    prompt_ko = f"{q_ko}? {ans_patterns['ko']}"
    prompt_it = f"{q_it}? {ans_patterns['it']}"
    
    # save as dict
    if c <= 2000:
        train['ja'].append((prompt_ja, a_ja))
        train['nl'].append((prompt_nl, a_nl))
        train['ko'].append((prompt_ko, a_ko))
        train['it'].append((prompt_it, a_it))
    elif c > 2000:
        test['ja'].append((prompt_ja, a_ja))
        test['nl'].append((prompt_nl, a_nl))
        test['ko'].append((prompt_ko, a_ko))
        test['it'].append((prompt_it, a_it))

    c += 1

# save as pkl.
path_train = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qa/train.pkl'
path_test = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/qa/test.pkl'
save_as_pickle(path_train, train)
save_as_pickle(path_test, test)