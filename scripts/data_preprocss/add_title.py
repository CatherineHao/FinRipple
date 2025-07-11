import os
import pandas as pd

directory = '/home/bld/data/data3/graph/datasets/instruction_result'

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        
        df = pd.read_csv(file_path, header=None)
        
        if df.iloc[0, 0] != 'instruction' or df.iloc[0, 1] != 'response':
            df = pd.concat([pd.DataFrame([['instruction', 'response']], columns=df.columns), df], ignore_index=True)
            df.to_csv(file_path, index=False, header=False)
            print(f"update file: {filename}")
        else:
            print(f"meet format: {filename}")

print("All files done!")