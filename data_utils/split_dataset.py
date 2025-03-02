import json
import random
from pathlib import Path
import os
from collections import Counter  

def case_dist(input_file: str):
    """
    Calculate the distribution of cases in the dataset.
    
    Args:
        input_file (str): Path to the input JSON file containing the dataset
        
    Returns:
        Counter: A counter object with case IDs as keys and their frequencies as values
    """
    with open(input_file, "r") as f:
        out = json.load(f)
    cases = [x[0]["video_filename"].split("_")[2] for x in out]
    return Counter(cases)

def split_dataset(input_file: str, train_ratio: float = 0.9, balance_train: bool = True):
    """
    Split a dataset into training and validation sets.
    
    Args:
        input_file (str): Path to the input JSON file containing the dataset
        train_ratio (float, optional): Ratio of data to use for training. Defaults to 0.9.
        balance_train (bool, optional): Whether to balance the training set by duplicating underrepresented cases. Defaults to True.
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
    
    # Balance the training set if requested
    if balance_train:
        # Get case distribution in the training set
        train_cases = [item[0]["video_filename"].split("_")[2] for item in train_data]
        case_counts = Counter(train_cases)
        
        # Find the most frequent case count
        max_count = max(case_counts.values())
        
        # Create a dictionary mapping each case to its samples
        case_to_samples = {}
        for item in train_data:
            case = item[0]["video_filename"].split("_")[2]
            if case not in case_to_samples:
                case_to_samples[case] = []
            case_to_samples[case].append(item)
        
        # Create balanced training data by duplicating underrepresented cases
        balanced_train_data = []
        for case, samples in case_to_samples.items():
            current_count = len(samples)
            duplication_factor = max_count // current_count
            
            # Add the original samples
            balanced_train_data.extend(samples)
            
            # Add duplicated samples
            for _ in range(duplication_factor - 1):
                balanced_train_data.extend(samples)
                
        # Shuffle the balanced training data
        random.shuffle(balanced_train_data)
        train_data = balanced_train_data
        
        print(f"Balanced training data: {len(train_data)} samples")
        print(f"Original case distribution: {dict(case_counts)}")
        print(f"After balancing, each case has approximately {max_count} samples")
    
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
    split_dataset(input_file, balance_train=True) 