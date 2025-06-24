from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from argparse import ArgumentParser

import numpy as np
import torch
import transformers
import wandb
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

from custom_trainer import (
    Trainer,
)
from auto_push_callback import AutoPushCallback
from funcs import (
    unfreeze_pickle,
)


if __name__ == '__main__':
    # login wandb and huggingface hub.
    load_dotenv() # load .env
    wandb.login(key=os.environ["WANDB_API_KEY"]) # wandb
    login(token=os.environ['HF_TOKEN_proj_LA']) # hf

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, help='model path on huggingface hub.', required=True)
    parser.add_argument('--lang', type=str, help='the L2 language you want to use as a dataset (L1=en, fixed).', required=True)
    parser.add_argument('--score_type', type=str, default='cos_sim', help='the type of distance fn used for identifying Transfer Neurons.')
    parser.add_argument('--top_n', type=int, default=1000, help='the number of neurons used for updating gradients per each type of Transfer Neurons.')

    args = parser.parse_args()
    L2 = args.lang
    model_name = args.model_name
    model_type = 'llama3' if 'llama' in model_name else 'mistral' if 'mistral' in model_name else 'aya' if 'aya' in model_name else 'bloom'
    score_type = args.score_type # 
    top_n = args.top_n # top-n neurons
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # add pad_token # llamaなどは'[PAD]' tokenをもっていないため.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    """ prepare datasets. """
    def tokenize(examples): # <- copied from: https://huggingface.co/docs/transformers/en/training
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    # train data
    monolingual_sentences = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_mono_train.pkl")
    dataset_train = Dataset.from_dict({'text': monolingual_sentences})
    tokenized_dataset_train = dataset_train.map(tokenize, batched=True)
    # test data
    monolingual_sentences = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_mono_test.pkl")
    dataset_test = Dataset.from_dict({'text': monolingual_sentences})
    tokenized_dataset_test = dataset_test.map(tokenize, batched=True)

    print("LOAD DATA FINISHED")

    """ prepare Transfer Neurons as required format. """
    def build_activate_neuron(neuron_list, module_name='mlp_down'):
        """
        Convert a list of (layer_i, neuron_i) into activate_neuron dictionary format.
        """
        activate_neuron = {module_name: {}}

        for layer_i, neuron_i in neuron_list:
            layer_key = str(layer_i)  # layer index as string
            if layer_key not in activate_neuron[module_name]:
                activate_neuron[module_name][layer_key] = set()
            activate_neuron[module_name][layer_key].add(neuron_i)

        return activate_neuron
    
    # type-1 neurons.
    tn_type1 = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl")
    tn_type1 = [neuron for neuron in tn_type1 if neuron[0] in [ _ for _ in range(20)]]
    tn_type1 = tn_type1[:top_n]
    # type-2 neurons.
    tn_type2 = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl")
    tn_type2 = [neuron for neuron in tn_type2 if neuron[0] in [ _ for _ in range(20, 32)]] if model_type in ['llama3', 'mistral', 'aya'] else [neuron for neuron in tn_type2 if neuron[0] in [ _ for _ in range(20, 30)]]
    tn_type2 = tn_type2[:top_n]
    # type1/2 neurons.
    tn = tn_type1 + tn_type2

    # get tn in required format.
    if model_type in ['llama3', 'mistral', 'aya']:
        tn = build_activate_neuron(tn, module_name='mlp_down') # module_name: the parameters we want to update.
    elif model_type in ['bloom']:
        # tn = build_activate_neuron(tn, module_name='dense_4h_to_h')
        tn = build_activate_neuron(tn, module_name='mlp_down')
    # path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/ft/tn_pt/{score_type}/{L2}.pt'
    # torch.save(tn, path)
    # sys.exit()

    """ implement training. """
    training_args = transformers.TrainingArguments(
        run_name=f'FT-TN-{L2}-{model_type}-{top_n}',
        per_device_train_batch_size=1,  # 4だと out of memory error.
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # gradientを何ステップ貯めてからパラメータを更新するか.
        warmup_ratio=0.03,
        num_train_epochs=1,
        learning_rate=5e-5, # この値がよく使われる?
        bf16=True, # for saving memory.
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        weight_decay=0.01,
        lr_scheduler_type='linear', # cosineの方がいい？
        logging_steps=10,
        optim='adamw_torch',
        eval_strategy='epoch',
        save_strategy='no', # localには保存しない.
        output_dir=f'./outputs/FT-TN-{L2}-{model_type}-{top_n}',
        save_total_limit=3,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=True,
        report_to='wandb',
        logging_nan_inf_filter=True,
        logging_first_step=True,
    )
    # set tn for training.
    training_args.activate_neuron = tn

    repo_name = f'HinataTezuka/FT-TN-{L2}-{model_type}-{top_n}' # huggingface repo.

    # for pushing the trained model to hugginface hub after every epoch.
    auto_push_cb = AutoPushCallback(
        push_repo_name=repo_name,
        tokenizer=tokenizer
    )

    # use CustomTrainer to train.
    trainer = Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_test,
        args=training_args,
        data_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8
        ),
        # callbacks=[auto_push_cb], # push trained model to huggigface hub per an epoch.
    )

    # begin training.
    trainer.train()

    # after training, push model to hub:
    trainer.model.push_to_hub(repo_name, tokenizer=tokenizer)