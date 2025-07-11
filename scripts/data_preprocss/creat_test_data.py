import os
import pandas as pd
import random

source_dir = ''
target_dir = ''

os.makedirs(target_dir, exist_ok=True)

for filename in os.listdir(source_dir):
    if filename.endswith('.csv'):
        source_path = os.path.join(source_dir, filename)
        
        df = pd.read_csv(source_path)
        
        if len(df) <= 20:
            selected_df = df
        else:
            selected_df = df.sample(n=20, random_state=random.randint(1, 10000))
        
        new_filename = f"{os.path.splitext(filename)[0]}_select_20.csv"
        target_path = os.path.join(target_dir, new_filename)
        
        selected_df.to_csv(target_path, index=False)
        
        print(f"Done: {filename} -> {new_filename}")

print("All files done!")