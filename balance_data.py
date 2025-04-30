import pandas as pd
import numpy as np
from pathlib import Path

def balance_dataset(input_file, output_file):
    print(f"Processing {input_file}...")
    
    # Read the data in chunks to handle large files
    chunks = []
    for chunk in pd.read_csv(input_file, chunksize=100000):
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Count clicked and non-clicked samples
    clicked_count = df['clicked'].sum()
    print(f"Total samples: {len(df)}")
    print(f"Clicked samples: {clicked_count}")
    print(f"Non-clicked samples: {len(df) - clicked_count}")
    
    # Separate clicked and non-clicked samples
    clicked_df = df[df['clicked'] == 1]
    non_clicked_df = df[df['clicked'] == 0]
    
    # Randomly sample non-clicked samples to match clicked count
    non_clicked_sampled = non_clicked_df.sample(n=clicked_count, random_state=42)
    
    # Combine clicked and sampled non-clicked data
    balanced_df = pd.concat([clicked_df, non_clicked_sampled], ignore_index=True)
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the balanced dataset
    balanced_df.to_csv(output_file, index=False)
    print(f"Balanced dataset saved to {output_file}")
    print(f"New dataset size: {len(balanced_df)}")
    print(f"New clicked count: {balanced_df['clicked'].sum()}")
    print(f"New non-clicked count: {len(balanced_df) - balanced_df['clicked'].sum()}\n")

def main():
    # Process training data
    balance_dataset('final_train.csv', 'balanced_train.csv')
    
    # Process test data
    balance_dataset('final_test.csv', 'balanced_test.csv')

if __name__ == "__main__":
    main() 