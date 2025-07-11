<div align="center">

### FinRipple: Aligning Large Language Models with Financial Market for Event Ripple Effect Awareness
</div>

<div align="center" style="display: flex; justify-content: center; align-items: center; gap: 20px;">

<a href="https://arxiv.org/abs/2505.23826" style="display: flex; align-items: center;">
  <img src="https://img.shields.io/badge/arxiv-red" alt="arxiv" style="height: 20px; vertical-align: middle;">
</a>

<!-- <a href="src/finripple_poster0701.pdf" style="display: flex; align-items: center;">
  <img src="https://img.shields.io/badge/View%20Poster-PDF-red?style=for-the-badge&logo=adobe-acrobat-reader" alt="View Poster PDF" style="height: 20px; vertical-align: middle;">
</a> -->

</div>

[![Poster Preview](src/finripple_poster0701.pdf)](src/finripple_poster0701.pdf)
---

### Overview
FinRipple is a comprehensive training framework designed for financial event analysis. This framework enables fine-tuning of large language models (LLMs) with financial news using techniques such as Proximal Policy Optimization (PPO) and parameter-efficient fine-tuning (PEFT) adapters. The goal is to enhance the model's ability to understand and generate responses in financial contexts effectively.

The project structure includes data preprocessing, training, post-processing, and evaluation components, all organized into modular scripts. Note that some paths and components have been removed in this release, and a complete version will be published soon.

---

### Project Structure

#### Root Directory
- **data/**: Contains training and evaluation datasets.
- **peft/**: Holds files and scripts related to Parameter-Efficient Fine-Tuning (PEFT) configurations.
- **scripts/**: Directory containing all the Python scripts for various stages of data handling and model training.
- **main_ppo**: Entry point for starting the PPO-based training of the model.
- **run_inference_base**: Script for running inference on the base model to evaluate the performance.

#### scripts/ Directory
This directory contains multiple subdirectories, each designed for specific functions within the training pipeline:

##### scripts/data_preprocess/
- **add_title.py**: Script to add titles to the dataset where necessary.
- **create_test_data.py**: Script to create test datasets from the original data, used for validation and testing purposes.
- **metric_script.py**: Script to calculate relevant metrics on preprocessed data for evaluation purposes.
- **prompt.py**: Contains methods to generate prompts from the news data to be used during training.

##### scripts/data_postprocess/
- **jsons/**: Folder containing processed JSON files.
- **news/**: Contains news articles in text format, used for post-training evaluations.
- **results/**: Stores the results of model evaluations.
- **merge_json.py**: Script to merge multiple JSON files into a consolidated dataset.
- **preprocess_json.py**: Script for post-processing JSON data, adding any missing fields or formatting data as required.
- **timeseries.xlsx**: Excel file containing time-series data, used for evaluating model impact on financial metrics.

#### scripts/ Core Scripts
- **finetune.py**: Script for fine-tuning the base model with the preprocessed datasets.
- **train_adapter.py**: Script for training adapters using a specified dataset. Adapters are trained for efficient adaptation to the financial domain.
- **train_backbone.py**: Script for training the main model backbone using conventional supervised learning.
- **inference.py**: Script for performing inference using the trained model to generate predictions based on financial news.
- **evaluation.py**: Script for evaluating the trained model's performance on specific metrics, especially in a financial context.

### How to Run
1. **Data Preparation**: Place your data in the `data/` directory and run the scripts in `scripts/data_preprocess/` for data cleaning and formatting.
2. **Training**: Use `main_ppo` to start PPO training or `train_adapter.py` and `train_backbone.py` for adapter or full model training, respectively.
3. **Inference and Evaluation**:
   - Run inference using `inference.py` with the trained model to generate responses.
   - Use `evaluation.py` to evaluate the model's outputs based on metrics suitable for financial analysis.

### Core Code Description
#### `train_ppo.py`
The `train_ppo.py` script contains the PPO training routine. The training involves:
- **Loading Data**: Loads preprocessed financial news data for use during training.
- **Training Loop**: Uses the PPO algorithm to fine-tune the model. Rewards are computed based on model output and compared with the desired output to improve training efficiency.

#### `train_adapter.py`
- **Adapter Training**: This script fine-tunes adapters using a smaller subset of data for efficient transfer learning in the financial domain.

#### `finetune.py`
- **Fine-tuning**: The script is used to train the base model using conventional supervised learning, taking full datasets to fine-tune weights to the financial data specifics.

#### `inference.py`
- **Inference and Testing**: Used to run the trained model on unseen data. This script helps evaluate how well the model adapts to the financial domain when given new financial news.

### Future Releases
- The paths have been partially removed in this version for clarity. We will provide a complete and structured framework for end-to-end training and evaluation in future releases, including detailed configuration files and example datasets.

### Data Example
- Example data files can be found in `scripts/data_preprocess/` and `scripts/data_postprocess/` to help with understanding the input-output structure expected by each script.

---
### Citation

If this work is helpful, please kindly cite as:

```bibtex
@misc{xu2025finripple,
      title={FinRipple: Aligning Large Language Models with Financial Market for Event Ripple Effect Awareness}, 
      author={Yuanjian Xu and Jianing Hao and Kunsheng Tang and Jingnan Chen and Anxian Liu and Peng Liu and Guang Zhang},
      year={2025},
      eprint={2505.23826},
      archivePrefix={arXiv},
      primaryClass={cs.SI},
      url={https://arxiv.org/abs/2505.23826}, 
}
```

---
### Contact and Contributions
If you have any questions or would like to contribute, please feel free to reach out. Contributions are welcome, especially in the areas of optimizing training routines, new model adapters, and expanding datasets.

