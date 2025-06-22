import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('')