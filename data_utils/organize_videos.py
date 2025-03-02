import json
import os
import shutil
from pathlib import Path
from glob import glob

def organize_videos():
    """
    Organize videos into train, validation, and test directories based on JSON files.
    
    This function:
    1. Reads train and validation JSON files to identify which videos belong to each set
    2. Creates train, validation, and test directories
    3. Moves videos to their respective directories
    4. Prints a summary of the organization process
    """
    # Define paths
    base_dir = "datasets/cataract1k/"
    videos_dir = os.path.join(base_dir, "videos")
    train_json = os.path.join(base_dir, "train_qa_pairs.json")
    val_json = os.path.join(base_dir, "val_qa_pairs.json")
    
    # Create train, validation, and test directories if they don't exist
    train_dir = os.path.join(base_dir, "videos/train")
    test_dir = os.path.join(base_dir, "videos/test")
    val_dir = os.path.join(base_dir, "videos/val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Load JSON files containing train and validation data
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    
    # Extract video filenames from JSON data
    train_videos = set()
    val_videos = set()
    
    # Process train data to extract video filenames
    for qa_group in train_data:
        for qa_pair in qa_group:
            if "video_filename" in qa_pair:
                train_videos.add(qa_pair["video_filename"])
    
    # Process validation data to extract video filenames
    for qa_group in val_data:
        for qa_pair in qa_group:
            if "video_filename" in qa_pair:
                val_videos.add(qa_pair["video_filename"])
    
    # Initialize counters and list for tracking moved videos
    moved_count = {"train": 0, "val": 0, "test": 0}
    not_found = []

    # Move videos to their respective directories
    for video in glob(f"{videos_dir}/*.mp4"):
        video_name = video.split("/")[-1]
        
        # Move to train directory if in train set
        if video_name in train_videos:
            shutil.move(video, os.path.join(train_dir, video_name))
            moved_count["train"] += 1
        # Move to validation directory if in validation set
        elif video_name in val_videos:
            shutil.move(video, os.path.join(val_dir, video_name))
            moved_count["val"] += 1
        # Move to test directory if not in train or validation sets
        else:
            shutil.move(video, os.path.join(test_dir, video_name))
            moved_count["test"] += 1
    
    # Print summary of the organization process
    print(f"Moved {moved_count['train']} videos to train folder")
    print(f"Moved {moved_count['val']} videos to val folder")
    print(f"Moved {moved_count['test']} videos to test folder")
    
    # Print warning if any videos were not found
    if not_found:
        print(f"\nWarning: {len(not_found)} videos were not found in either JSON file:")
        for video in not_found[:10]:  # Show first 10 only
            print(f"- {video}")
        if len(not_found) > 10:
            print(f"... and {len(not_found) - 10} more")

if __name__ == "__main__":
    organize_videos() 