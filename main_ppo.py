import logging
from peft import PeftModel, PeftConfig,LoraConfig
from scripts.train_ppo import train_ppo
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch
import os

def train_ppo_standalone(base_model_path, adapter_save_path, news_data_path, ppo_model_save_path,num_epochs=1, learning_rate=1e-5, batch_size=1):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'
    
    logging.info(f"Loading base model from '{base_model_path}'")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto"
    )
    
    logging.info(f"Loading adapter from '{adapter_save_path}'")
    peft_config = PeftConfig.from_pretrained(adapter_save_path)
    adapter_model = PeftModel.from_pretrained(base_model, adapter_save_path)
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    adapter_model = PeftModel(base_model, peft_config)
    adapter_model.config.use_cache = False
    adapter_model.config.pretraining_tp = 1

    logging.info(f"Loading tokenizer from '{base_model_path}'")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    adapter_model.resize_token_embeddings(len(tokenizer))

    # Turn PeftModel to AutoModelForCausalLMWithValueHead
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model,adapter_model)

    logging.info("Preparing model for training")
            
    # Unfreeze base model parameters for PPO training
    for param in ppo_model.parameters():
        param.requires_grad = True

    # Freeze adapter parameters during PPO training
    for name, param in adapter_model.named_parameters():
        if 'lora' in name:
            param.requires_grad = False

    # Run PPO training
    logging.info("Starting PPO training")
    train_ppo(
        model=ppo_model,
        tokenizer=tokenizer,
        news_data_path=news_data_path,
        ppo_model_save_path=ppo_model_save_path,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    logging.info("PPO training completed")

    # Save the PPO-trained model
    adapter_model.save_pretrained(ppo_model_save_path)
    logging.info(f"Model saved at {ppo_model_save_path}")
    return adapter_model

if __name__ == '__main__':
    base_model_path = 'gemma-2b-it'
    adapter_save_path = 'finetune_adapter/gemma-2b-it/merged_2020_01/checkpoint-20'
    news_data_path = 'data/news'
    ppo_model_save_path = 'PPO_models/gemma-2b-it'

    train_ppo_standalone(
        base_model_path=base_model_path,
        adapter_save_path=adapter_save_path,
        news_data_path=news_data_path,
        ppo_model_save_path=ppo_model_save_path,
        num_epochs=1,
        learning_rate=1e-5,
        batch_size=1
    )