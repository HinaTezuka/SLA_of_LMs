import os
import torch
import json
from typing import List, Optional
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
#import wandb
from fire import Fire
from dotenv import dotenv_values
from huggingface_hub import login

def train(
    # model/data params
    base_model: str = "",
    new_model:str = "", 
    train_data_path: str = "",
    valid_data_path: str = "",
    valid_split_name: str = "validation",
    prompt_template: str = "",
    load_in_8bit: bool = True,
    output_dir: str = "./logs",
    translation_task: bool = True,
    continuous_correction: bool = False,
    saved_full_model_path: Optional[str] = None, # load the full saved peft model
    # training hyperparams
    cutoff_len: int = 256,
    ################################################################################
    # QLoRA parameters
    ################################################################################
    lora_r: int = 32, # LoRA attention dimension
    lora_alpha: int = 16, # Alpha parameter for LoRA scaling
    lora_dropout: float = 0.1, # Dropout probability for LoRA layers
    ################################################################################
    # bitsandbytes parameters
    ################################################################################
    use_4bit: bool = True,# Activate 4-bit precision base model loading
    bnb_4bit_compute_dtype: str = "float16", # Compute dtype for 4-bit base models
    bnb_4bit_quant_type: str = "nf4",  # Quantization type (fp4 or nf4)
    use_nested_quant :bool = False, # Activate nested quantization for 4-bit base models (double quantization)
    ################################################################################
    # TrainingArguments parameters
    ################################################################################
    num_train_epochs: int = 5, # Number of training epochs tuning 4.6=4651/1004
    fp16: bool = False,
    bf16: bool = False, # set bf16 to True with an A100
    per_device_train_batch_size: int = 2, # Batch size per GPU for training
    per_device_eval_batch_size: int = 2, # Batch size per GPU for evaluation
    gradient_accumulation_steps: int = 2, # Number of update steps to accumulate the gradients for
    # per_device_train_batch_size * device_num * gradient_accumulation_steps = batch_sizeのはず
    gradient_checkpointing: bool = True, # Enable gradient checkpointing
    max_grad_norm: int = 0.3, # Maximum gradient normal (gradient clipping)
    learning_rate: float = 5e-5, # Initial learning rate (AdamW optimizer)
    weight_decay: float = 0.001, # Weight decay to apply to all layers except bias/LayerNorm weights
    optim: str = "paged_adamw_32bit", # Optimizer to use
    lr_scheduler_type: str = "linear", # "cosine" # Learning rate schedule
    max_steps: int = -1, # Number of training steps (overrides num_train_epochs)
    warmup_ratio: float = 0.03, # Ratio of steps for a linear warmup (from 0 to learning rate)
    group_by_length: bool = True, # Group sequences into batches with same length, Saves memory and speeds up training considerably
    save_strategy: str = "steps",
    evaluation_strategy: str = "steps",
    save_steps: int = 200, # Save checkpoint every X updates steps
    eval_steps: int = 100, # When load_best_model_at_end set to True, the parameters save_strategy needs to be the same as evaluation_strategy, and in the case it is “steps”, save_steps must be a round multiple of eval_step
    save_total_limit: int = 3,
    load_best_model_at_end: bool =True, # store best model on evaluation score 
    logging_steps = 25, # Log every X updates steps
    ################################################################################
    # SFT parameters
    ################################################################################
    max_seq_length = None, # Maximum sequence length to use
    packing:bool = False, # Pack multiple short examples in the same input sequence to increase efficiency
    device_map: str = "auto", # Load gpu setting on ABCI
    ################################################################################
    # huggingface parameters
    ################################################################################
    upload_to_huggingface: bool = False,
    ################################################################################
    # wandb parameters
    ################################################################################
    use_wandb: bool = False,
    wandb_project: str = "LET_sft",
    wandb_run_name: str = "default_run",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    ):
    '''
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
        )
    '''
    assert os.path.isfile(prompt_template), f'cannot locate {prompt_template}'

    # Load data
    train_data = load_dataset(train_data_path, split="train")
    val_data = load_dataset(valid_data_path, split=valid_split_name)
    # Instruction tuning
    def formatting_prompts_func_for_solver(example):
        with open(prompt_template) as f:
            prompt_dict = json.load(f)
            output_texts = []
            for i in range(len(example['conclusion'])):
                text = prompt_dict['input_template'].format(premises=example['premises'][i], conclusion=example['conclusion'][i], label=example['label'][i])
                output_texts.append(text)
            return output_texts

    def formatting_prompts_func_for_L2T(example):
        with open(prompt_template) as f:
            prompt_dict = json.load(f)
            output_texts = []
            for i in range(len(example['logical_form'])):
                text = prompt_dict['input_template'].format(logical_form=example['logical_form'][i], NL=example['original'][i])
                output_texts.append(text)
            return output_texts    

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype) 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load base model
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        #load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # 学習範囲を応答部分に限定
    response_template = "### English:"

    added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
    initial_token_count = len(tokenizer)
    model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)  


    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        report_to="wandb" if use_wandb else "tensorboard",
        run_name=wandb_run_name if use_wandb else None,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func_for_solver if 'solver' in prompt_template else formatting_prompts_func_for_L2T,    
        packing=packing, #formatting_func使うならfalseにする
        data_collator=collator, # 学習対象を回答部分に限定する v2
        args=training_arguments,
        #compute_metrics=compute_metrics
    )
    # Train model
    trainer.train()
    # Save trained model
    trainer.model.save_pretrained(new_model)

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Reload model in FP16 and merge it with LoRA weights
    base_model = LlamaForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()
    model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)

    if upload_to_huggingface:
        config = dotenv_values(".env")
        login(token=config['HUGGINGFACE_TOKEN_W'], write_permission=True)

        model.push_to_hub(new_model, use_temp_dir=False)
        tokenizer.push_to_hub(new_model, use_temp_dir=False)
    
    '''
    if use_wandb:
        wandb.finish()
    '''

if __name__ == "__main__":
    Fire(train)