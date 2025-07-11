import torch
import os
import pandas as pd
from datasets import load_dataset, Dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)

class EpochCallback(TrainerCallback):
    """Custom callback to track and print average loss after each epoch."""

    def __init__(self):
        self.running_loss = 0.0
        self.steps = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Logs loss value and accumulates it for each step."""
        if logs is not None and 'loss' in logs:
            self.running_loss += logs['loss']
            self.steps += 1

    def on_epoch_end(self, args, state, control, **kwargs):
        """Prints the average loss at the end of each epoch."""
        if self.steps > 0:
            avg_loss = self.running_loss / self.steps
            print(f"Epoch {state.epoch} finished. Average loss on this epoch: {avg_loss:.4f}")
            self.running_loss = 0.0
            self.steps = 0
        else:
            print(f"Epoch {state.epoch} finished. No loss data available.")

def format_data(example, model_type: str) -> dict:
    """
    Formats data according to the specified model type for training.

    Args:
        example (dict): A dictionary representing a single training example.
        model_type (str): Type of the model to determine the formatting.

    Returns:
        dict: A formatted dictionary containing the 'text' field.
    """
    if model_type == "llama2":
        return {
            "text": f"<s>[INST] {example['instruction']}[/INST] {example['response']}</s>"
        }
    elif model_type == "llama3":
        return {
            "text": f"<s>[INST] {example['instruction_en']}[/INST] {example['response_en']}</s>"
        }
    elif model_type == "gemma":
        return {
            "text": f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n<start_of_turn>model\n{example['response']}<end_of_turn>\n"
        }
    elif model_type == "phi":
        return {
            "text": f"<|user|>\n{example['instruction_en']}<|end|>\n<|assistant|>\n{example['response_en']}<|end|>\n"
        }
    elif model_type == "vicuna":
        return {
            "text": f"USER: {example['instruction_en']}\nASSISTANT: {example['response_en']}</s>\n"
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def sft_pipeline(base_model_path: str, train_file: str, output_path: str, model_type: str):
    """
    Fine-tunes a model using Supervised Fine-Tuning (SFT) pipeline.

    Args:
        base_model_path (str): Path to the pretrained base model.
        train_file (str): Path to the training data in JSON format.
        output_path (str): Path to save the fine-tuned model.
        model_type (str): The model type for data formatting.
    """
    # Determine computation settings based on GPU capability
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'

    # Load training data and format
    train_df = pd.read_json(train_file)
    train_dataset = Dataset.from_pandas(train_df).map(lambda x: format_data(x, model_type))

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.resize_token_embeddings(len(tokenizer))

    # Configure PEFT (Parameter-Efficient Fine-Tuning)
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    # Set up output directory
    output_folder = os.path.basename(train_file).split('_result')[0]
    output_dir = os.path.join(output_path, output_folder)

    # Set training arguments
    training_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=16,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        logging_steps=1,
        learning_rate=1e-3,
        weight_decay=0.0001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.5,
        max_steps=-1,
        warmup_ratio=0.3,
        group_by_length=True,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=999999999,
        gradient_checkpointing=True,
    )

    # Set up trainer with the model, dataset, and training arguments
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
        callbacks=[EpochCallback()]
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    # Execute the training pipeline
    sft_pipeline(
        base_model_path='',
        train_file='',
        output_path='',
        model_type='gemma'
    )
