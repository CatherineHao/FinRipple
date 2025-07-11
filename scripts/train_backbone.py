# src/scripts/train_ppo.py

import os
import logging
from trl import PPOTrainer, PPOConfig
from datasets import Dataset
import torch
from tqdm import tqdm
from accelerate import Accelerator
from scripts.prompt import PromptGen
import torch.nn.functional as F

def train_ppo(model, tokenizer, news_data_path, ppo_model_save_path, num_epochs=1, learning_rate=1e-5, batch_size=1):
    """
    Train the model using Proximal Policy Optimization (PPO) with news data.

    Args:
        model: The model to be trained.
        tokenizer: The tokenizer corresponding to the model.
        news_data_path (str): Path to the directory containing news data in text files.
        ppo_model_save_path (str): Path to save the PPO-trained model.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for PPO training.
        batch_size (int): Batch size for PPO training.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load news data and generate instructions
    logger.info(f"Loading news data from {news_data_path}")
    news_texts = []
    prompt = PromptGen()
    
    # Traverse through the directory structure to find text files
    for root, dirs, files in os.walk(news_data_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    instructions = prompt.generate_instructions(f.read())  # Generate instructions from the text file
                    news_texts.append(instructions)
    
    logger.info(f"Loaded {len(news_texts)} news articles")

    # Prepare dataset
    dataset = Dataset.from_dict({'text': news_texts})

    # Initialize Accelerator for managing GPU/CPU usage
    accelerator = Accelerator()

    # Configure PPO settings
    config = PPOConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        log_with=None,
        mini_batch_size=1,  
        gradient_accumulation_steps=1  # gradient_accumulation_steps * mini_batch_size = batch_size
    )

    # Initialize PPO Trainer
    logger.info("Initializing PPO Trainer")
    ppo_trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    # Prepare the model and optimizer with accelerator
    model, ppo_trainer.optimizer = accelerator.prepare(model, ppo_trainer.optimizer)

    # Start PPO training
    logger.info("Starting PPO training")
    for epoch in range(num_epochs):
        logger.info(f"Starting PPO epoch {epoch + 1}/{num_epochs}")
        
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Prepare a batch of input texts
            batch = dataset[i:i + batch_size]['text']
            unwrapped_model = accelerator.unwrap_model(model)

            # Tokenize the batch of queries
            query_tensors = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(accelerator.device)

            input_ids = query_tensors.input_ids
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            # Generate model responses
            response_ids = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=1024,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode generated responses
            response_tensors = [response.squeeze() for response in response_ids]
            generated_responses = [tokenizer.decode(response, skip_special_tokens=True) for response in response_ids]

            # Log the generated response
            for prompt, response in zip(batch, generated_responses):
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Generated text: {response}")
                logger.info("-" * 50)

            # Calculate rewards using the reward function
            Z_t, e_t = postprocess_responses(response_tensors)
            rewards = calculate_reward(Z_t, e_t, lambd=0.1)

            # Prepare query tensor list for PPO
            query_tensor_list = [tensor for tensor in query_tensors.input_ids]

            # PPO step
            ppo_trainer.step(query_tensor_list, response_tensors, rewards)

            # Calculate average reward and log it
            avg_reward = sum(reward.item() for reward in rewards) / len(rewards)
            logger.info(f"Batch {i // batch_size + 1}, Average Reward: {avg_reward:.4f}")

    # Save the PPO-trained model
    logger.info(f"Saving PPO-trained model to {ppo_model_save_path}")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(ppo_model_save_path)


def calculate_reward(Z_t, e_t, lambd=0.1):
    """
    Calculates the reward at time t based on the given inputs.

    Args:
        Z_t (torch.Tensor): Tensor representing Z^t.
        e_t (torch.Tensor): Tensor representing e^t.
        lambd (float): Lambda value used in the reward calculation.

    Returns:
        torch.Tensor: Calculated reward value.
    """
    # Dot product of Z^t and e^t
    dot_product = torch.sum(Z_t * e_t, dim=1)

    # Norms of Z^t and e^t
    norm_Z_t = torch.norm(Z_t, dim=1)
    norm_e_t = torch.norm(e_t, dim=1)

    # First term: Z^t · e^t / (||Z^t|| ||e^t||)
    first_term = dot_product / (norm_Z_t * norm_e_t)

    # Sum of min(Z^t_i, e^t_i) for each element
    min_sum = torch.sum(torch.min(Z_t, e_t), dim=1)

    # L1 norm of e^t
    l1_norm_e_t = torch.norm(e_t, p=1, dim=1)

    # Second term: lambda * (sum of min(Z^t_i, e^t_i) / ||e^t||_1)
    second_term = lambd * (min_sum / l1_norm_e_t)

    # Total reward
    reward = first_term + second_term

    return reward

if __name__ == "__main__":
    # Example usage of train_ppo function
    # Replace `model`, `tokenizer`, `news_data_path`, and `ppo_model_save_path` with actual paths/models.

    # Example configuration
    base_model = None  # Replace with the actual model
    tokenizer = None   # Replace with the actual tokenizer
    news_data_path = 'path/to/news_data'  # Replace with actual news data path
    ppo_model_save_path = 'path/to/save/ppo_model'  # Replace with actual PPO model save path

    train_ppo(
        model=base_model,
        tokenizer=tokenizer,
        news_data_path=news_data_path,
        ppo_model_save_path=ppo_model_save_path,
        num_epochs=1,
        learning_rate=1e-5,
        batch_size=1
    )

