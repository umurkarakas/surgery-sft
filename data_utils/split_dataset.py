import json
import random
from pathlib import Path
import os

def split_dataset(input_file: str, train_ratio: float = 0.9):
    """
    Split a dataset into training and validation sets.
    
    Args:
        input_file (str): Path to the input JSON file containing the dataset
        train_ratio (float, optional): Ratio of data to use for training. Defaults to 0.9.
    """
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Shuffle the data with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(data)
    
    # Calculate split index based on train ratio
    split_idx = int(len(data) * train_ratio)
    
    # Split the data into train and validation sets
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Create output paths
    input_path = Path(input_file)
    output_dir = input_path.parent
    
    # Save train split to JSON file
    train_path = output_dir / 'train_qa_pairs.json'
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    # Save validation split to JSON file
    val_path = output_dir / 'val_qa_pairs.json'
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    # Print summary statistics
    print(f"Total samples: {len(data)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"\nFiles saved:")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")

if __name__ == "__main__":
    input_file = "datasets/cataract1k/qa_pairs_without_idle.json"
    split_dataset(input_file) 