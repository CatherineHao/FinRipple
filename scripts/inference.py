import torch
import os
import sys
import json
import pandas as pd
from datasets import load_dataset, Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

def format_data(prompt: str, model_type: str) -> str:
    """
    Formats the prompt based on the given model type.

    Args:
        prompt (str): The user's input prompt.
        model_type (str): Type of the model to determine formatting.

    Returns:
        str: Formatted input text.
    """
    if model_type == "llama2" or model_type == "llama3":
        return f"<s>[INST] {prompt} [/INST]"
    elif model_type == "gemma":
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    elif model_type == "phi":
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    elif model_type == "vicuna":
        return f"USER: {prompt}\nASSISTANT:"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_model_and_tokenizer(base_model_path: str, given_adapter_path: str, is_adapter: bool):
    """
    Loads the model and tokenizer for inference.

    Args:
        base_model_path (str): Path to the pretrained base model.
        given_adapter_path (str): Path to the adapter model, if applicable.
        is_adapter (bool): Flag indicating whether to use an adapter model.

    Returns:
        model, tokenizer: Loaded model and tokenizer.
    """
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto"
    )

    # If an adapter is provided, load it
    if is_adapter:
        model = PeftModel.from_pretrained(model, given_adapter_path)

    # Set model configuration
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def run_inference(model, tokenizer, prompts: pd.Series, model_type: str):
    """
    Runs inference on a list of prompts using the given model and tokenizer.

    Args:
        model: The pretrained model for inference.
        tokenizer: Tokenizer to encode the input.
        prompts (pd.Series): Series of prompts to generate responses.
        model_type (str): Type of the model for formatting.

    Returns:
        list of dict: Generated responses for each prompt.
    """
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    results = []
    for prompt in prompts:
        input_text = format_data(prompt, model_type)
        result = pipe(input_text, max_new_tokens=20, do_sample=True, top_p=1.0, temperature=0.1)
        generated_text = result[0]['generated_text']

        response = generated_text[len(input_text):].strip()
        results.append({'instruction_en': prompt, 'response_en': response})

        # Log prompt and response
        print(f"Prompt: {prompt}")
        print(f"Generated text: {response}")
        print("-" * 50)

    return results

def save_results(results: list, prompt_path: str, output_path: str):
    """
    Saves the generated responses to a JSON file.

    Args:
        results (list): List of dictionaries containing prompts and generated responses.
        prompt_path (str): Path to the prompt CSV to derive output file naming.
        output_path (str): Directory where the output JSON should be saved.
    """
    input_filename = os.path.basename(prompt_path)
    input_filename_without_ext = os.path.splitext(input_filename)[0]

    output_filename = f"{input_filename_without_ext}.json"
    output_file = os.path.join(output_path, output_filename)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_file}")

def inference_pipeline(base_model_path: str, given_adapter_path: str, prompt_path: str, model_type: str, output_path: str, is_adapter: str):
    """
    Main pipeline function for model inference.

    Args:
        base_model_path (str): Path to the pretrained base model.
        given_adapter_path (str): Path to the adapter model, if applicable.
        prompt_path (str): Path to the CSV file containing prompts.
        model_type (str): Type of the model to determine the prompt formatting.
        output_path (str): Directory where the output JSON should be saved.
        is_adapter (str): String flag indicating if an adapter is used ("True"/"False").
    """
    # Convert adapter flag to boolean
    is_adapter = True if is_adapter.lower() == "true" else False

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(base_model_path, given_adapter_path, is_adapter)

    # Load prompts from CSV
    df = pd.read_csv(prompt_path)

    # Run inference on prompts
    results = run_inference(model, tokenizer, df['instruction_en'], model_type)

    # Save results to output file
    save_results(results, prompt_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python script.py <model_path> <adapter_path> <prompt_path> <model_type> <output_path> <is_adapter>")
        sys.exit(1)

    base_model_path = sys.argv[1]
    adapter_path = sys.argv[2]
    prompt_path = sys.argv[3]
    model_type = sys.argv[4]
    output_path = sys.argv[5]
    is_adapter = sys.argv[6]

    inference_pipeline(base_model_path, adapter_path, prompt_path, model_type, output_path, is_adapter)
