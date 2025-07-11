# scripts/train_adapter.py

import os
import json
import logging
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

def load_knowledge_graph_data(kg_data_path: str) -> list:
    """
    Loads the knowledge graph data from the specified path.

    Args:
        kg_data_path (str): Path to the knowledge graph JSON file.

    Returns:
        list: A list of dictionaries containing the instructions and responses.
    """
    kg_data_file = os.path.join(kg_data_path, 'kg_data.json')
    with open(kg_data_file, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    return kg_data

def prepare_dataset(kg_data: list) -> Dataset:
    """
    Prepares a dataset from knowledge graph data.

    Args:
        kg_data (list): Knowledge graph data containing instructions and responses.

    Returns:
        Dataset: A HuggingFace Dataset containing instructions and responses.
    """
    instructions = [item['instruction'] for item in kg_data]
    responses = [item['response'] for item in kg_data]
    dataset = Dataset.from_dict({
        'instruction': instructions,
        'response': responses
    })
    return dataset

def preprocess_function(examples, tokenizer):
    """
    Preprocesses the dataset examples by tokenizing and concatenating inputs and outputs.

    Args:
        examples (dict): A dictionary containing dataset examples.
        tokenizer: The tokenizer to use for processing.

    Returns:
        dict: A dictionary of tokenized examples ready for training.
    """
    inputs = examples['instruction']
    outputs = examples['response']
    full_texts = [inp + tokenizer.eos_token + out for inp, out in zip(inputs, outputs)]
    tokenized = tokenizer(full_texts, max_length=1024, truncation=True)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

def train_adapter(base_model, tokenizer, kg_data_path, adapter_save_path,
                  adapter_method='lora', adapter_config=None,
                  num_epochs=3, learning_rate=1e-4, batch_size=8):
    """
    Trains the adapter with knowledge graph data.

    Args:
        base_model: The base language model.
        tokenizer: The tokenizer corresponding to the base model.
        kg_data_path (str): Path to the knowledge graph data.
        adapter_save_path (str): Path to save the trained adapter.
        adapter_method (str): Adapter method to use (e.g., 'lora').
        adapter_config (dict): Configuration for the adapter.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for training.
        batch_size (int): Batch size for training.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load knowledge graph data
    logger.info(f"Loading knowledge graph data from {kg_data_path}")
    kg_data = load_knowledge_graph_data(kg_data_path)

    # Prepare dataset
    dataset = prepare_dataset(kg_data)
    logger.info(f"Loaded {len(dataset)} examples from knowledge graph data")

    # Preprocess dataset
    logger.info("Tokenizing dataset")
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=['instruction', 'response'])

    # Set up PEFT configuration
    if adapter_method == 'lora':
        logger.info("Configuring LoRA adapter")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=adapter_config.get('r', 8),
            lora_alpha=adapter_config.get('alpha', 32),
            lora_dropout=adapter_config.get('dropout', 0.1)
        )
    else:
        raise NotImplementedError(f"Adapter method '{adapter_method}' is not implemented.")

    # Convert base model to PEFT model
    logger.info("Converting base model to PEFT model")
    peft_model = get_peft_model(base_model, peft_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=adapter_save_path,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_steps=100,
        logging_steps=10,
        learning_rate=learning_rate,
        fp16=True,
        save_total_limit=1,
        logging_dir=os.path.join(adapter_save_path, 'logs'),
        logging_strategy='steps',
        logging_steps=10
    )

    # Set up DataCollator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Initialize Trainer
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # Begin training
    logger.info("Starting adapter training")
    trainer.train()

    # Save the trained adapter
    logger.info(f"Saving trained adapter to {adapter_save_path}")
    peft_model.save_pretrained(adapter_save_path)

if __name__ == "__main__":
    # Example usage of the train_adapter function
    # Replace `base_model`, `tokenizer`, `kg_data_path`, and `adapter_save_path` with actual paths/models.

    # Example configuration
    base_model = None  # Replace with the actual model
    tokenizer = None   # Replace with the actual tokenizer
    kg_data_path = 'path/to/knowledge_graph/data'  # Replace with actual knowledge graph data path
    adapter_save_path = 'path/to/save/adapter'     # Replace with actual adapter save path

    # Adapter configuration
    adapter_config = {
        'r': 8,
        'alpha': 32,
        'dropout': 0.1
    }

    train_adapter(
        base_model=base_model,
        tokenizer=tokenizer,
        kg_data_path=kg_data_path,
        adapter_save_path=adapter_save_path,
        adapter_method='lora',
        adapter_config=adapter_config,
        num_epochs=3,
        learning_rate=1e-4,
        batch_size=8
    )
