import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from collections import defaultdict

from datasets import load_dataset

from funcs import (
    save_as_pickle,
    unfreeze_pickle,
)

langs = ["ja", "nl", "ko", "it", "en"]
# load QA dataset.
qa_num = 1000
qa = load_dataset('apple/mkqa')['train']
qa = qa.shuffle(seed=42)

sentences = defaultdict(list)
c = 0
for i in range(len(qa['queries'])):
    if c == qa_num: break
    txt_ja = qa['queries'][i]['ja']
    txt_nl = qa['queries'][i]['nl']
    txt_ko = qa['queries'][i]['ko']
    txt_it = qa['queries'][i]['it']
    txt_en = qa['queries'][i]['en']

    if any(not txt for txt in [txt_ja, txt_nl, txt_ko, txt_it, txt_en]):
        continue
    
    sentences['ja'].append(txt_ja)
    sentences['nl'].append(txt_nl)
    sentences['ko'].append(txt_ko)
    sentences['it'].append(txt_it)
    sentences['en'].append(txt_en)
    
    c += 1

path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/mkqa_q_sentence_data_ja_nl_ko_it_en.pkl'
save_as_pickle(path, dict(sentences))