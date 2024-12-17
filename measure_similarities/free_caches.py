"""
GPT2をロードしたメモリを解放（GPT2-smallをローカルで検証した場合。）
"""
import sys
sys.path.append('../gpt2')

from gpt2_measure_similarity import *

import gc

del gpt2_model_original
del gpt2_model_ja
del gpt2_model_du
del gpt2_model_ger
del gpt2_model_ita
del gpt2_model_fre
del gpt2_model_ko
del gpt2_model_spa

gc.collect()
