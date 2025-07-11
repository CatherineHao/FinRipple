import json
import csv
import pandas as pd
import numpy as np
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import nltk
import os

nltk.download('punkt')

def calculate_perplexity(text, ngram=2):
    tokens = nltk.word_tokenize(text.lower())
    train, vocab = padded_everygram_pipeline(ngram, [tokens])
    lm = MLE(ngram)
    lm.fit(train, vocab)
    test = tokens
    return lm.perplexity(test)

def normalize_perplexity(perplexity):
    normalized = 1 - (np.log(perplexity) / np.log(1e6))  
    return max(0, min(normalized, 1))

def compare_responses(csv_file, json_file):
    df_csv = pd.read_csv(csv_file)
    
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    results = []
    total_accuracy = 0
    
    for json_item in json_data:
        csv_response = df_csv[df_csv['instruction_en'] == json_item['instruction_en']]['response_en'].iloc[0]
        json_response = json_item['response_en']
        
        csv_perplexity = calculate_perplexity(csv_response)
        json_perplexity = calculate_perplexity(json_response)
        
        accuracy = normalize_perplexity(json_perplexity) / normalize_perplexity(csv_perplexity)
        accuracy_percentage = accuracy * 100
        
        results.append({
            'instruction_en': json_item['instruction_en'],
            'csv_response': csv_response,
            'json_response': json_response,
            'accuracy': f'{accuracy_percentage:.2f}%'
        })
        
        total_accuracy += accuracy
    
    avg_accuracy = (total_accuracy / len(json_data)) * 100
    
    return results, avg_accuracy

def main(csv_file, json_file):
    results, avg_accuracy = compare_responses(csv_file, json_file)
    
    output_file = json_file.replace('_result_select_20.json', '_accuracy.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['instruction_en', 'csv_response', 'json_response', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Individual accuracies saved to {output_file}")
    
    avg_output_file = json_file.replace('_result_select_20.json', '.csv')
    pd.DataFrame({'Avg Accuracy': [f'{avg_accuracy:.2f}%']}).to_csv(avg_output_file, index=False)
    
    print(f"Average accuracy saved to {avg_output_file}")

if __name__ == "__main__":
    csv_file = '/home/bld/data/data3/graph/datasets/test_data/raw_data/merged_1980_01_result_select_20.csv'  
    json_file = '/home/bld/data/data3/graph/inference_results/ft_models/Llama-2-13b-chat-hf/merged_1980_01_result_select_20.json' 
    main(csv_file, json_file)