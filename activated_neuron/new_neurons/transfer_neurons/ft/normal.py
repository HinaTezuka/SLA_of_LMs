import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from argparse import ArgumentParser

import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from custom_trainer import (
    Trainer,
)


if __name__ == '__main__':
    # parser = ArgumentParser()

    # parser.add_argument('')

    """ test """
    model_name = 'meta-llama/Meta-Llama-3-8B'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print(model.model.layers[0].mlp.down_proj.weight)
    print(model.model.layers[0].mlp.down_proj.weight.shape)
    sys.exit()