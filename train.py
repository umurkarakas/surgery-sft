import os
import json
import torch
import random
import argparse
import decord
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2_5_VLProcessor,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer, SFTConfig

# Initialize decord for video reading
# Set bridge to torch to ensure compatibility with PyTorch tensors
decord.bridge.set_bridge('torch')

class JSONLDataset(Dataset):
    """
    Custom dataset class for loading and processing video data with associated QA pairs.
    Inherits from PyTorch's Dataset class for compatibility with DataLoader.
    """
    def __init__(self, jsonl_file_path: str, video_directory_path: str):
        """
        Initialize the dataset with paths to the JSON file and video directory.
        
        Args:
            jsonl_file_path: Path to the JSON file containing QA pairs
            video_directory_path: Path to the directory containing video files
        """
        self.jsonl_file_path = jsonl_file_path
        self.video_directory_path = video_directory_path
        self.entries = self._load_entries()

    def _load_entries(self):
        """
        Load and parse the JSON file containing QA pairs.
        
        Returns:
            List of flattened entries with video filename and QA pairs
        """
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            data = json.load(file)
            for entry_list in data:
                # Flatten the nested structure for easier access
                flattened_entry = {
                    "video_filename": entry_list[0]["video_filename"],
                    "question1": entry_list[1]["question1"],
                    "answer1": entry_list[1]["answer1"],
                    "question2": entry_list[2]["question2"],
                    "answer2": entry_list[2]["answer2"],
                    "question3": entry_list[3]["question3"],
                    "answer3": entry_list[3]["answer3"]
                }
                entries.append(flattened_entry)
        return entries

    def __len__(self):
        """Return the number of entries in the dataset"""
        return len(self.entries)

    def __getitem__(self, idx: int):
        """
        Get a specific item from the dataset by index.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple containing video reader object, raw entry, and formatted data
        """
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        video_path = os.path.join(self.video_directory_path, entry['video_filename'])
        
        # Load video using decord
        vr = decord.VideoReader(video_path)
        
        return vr, entry, format_data(self.video_directory_path, entry)

def format_data(video_directory_path, entry):
    """
    Format the data into the chat template expected by the model.
    
    Args:
        video_directory_path: Path to the directory containing video files
        entry: Dictionary containing QA pairs and video filename
        
    Returns:
        List of dictionaries representing a conversation in the format expected by the model
    """
    # Define system message that instructs the model on its role and task
    SYSTEM_MESSAGE = """You are a vision language model specialized in analyzing cataract surgery videos.
Your task is to analyze the provided surgical video frames and extract relevant clinical information from:
- Current surgical phase/step
- Visible surgical instruments in use
- Visible and relevant anatomical structures

Focus on providing accurate, clinically relevant observations using proper medical terminology.
Base your responses solely on what is visible in the video frames.
Provide clear, concise answers."""

    # Format the conversation as a list of message dictionaries
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_directory_path + "/" + entry["video_filename"],
                },
                {
                    "type": "text",
                    "text": entry["question1"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": entry["answer1"]}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": entry["question2"]}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": entry["answer2"]}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": entry["question3"]}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": entry["answer3"]}],
        },
    ]

def sample_images(vr, sample_fps=2):
    """
    Sample frames from a video at a specified frame rate.
    
    Args:
        vr: VideoReader object from decord
        sample_fps: Target frames per second for sampling
        
    Returns:
        Tensor containing sampled video frames
    """
    num_frames = vr._num_frame
    # Calculate frame indices to sample based on the target FPS
    frames_idx = [int(vr.get_avg_fps() / sample_fps)*i for i in range(sample_fps * num_frames // int(vr.get_avg_fps()))] 
    return vr.get_batch(frames_idx)

def collate_fn(batch):
    """
    Custom collate function for DataLoader to process a batch of samples.
    
    Args:
        batch: List of tuples (vr, entry, example) from the dataset
        
    Returns:
        Dictionary of model inputs including tokenized text and processed video frames
    """
    vrs, _, examples = zip(*batch)

    # Apply chat template to format conversations
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    # Sample frames from each video
    video_inputs = [sample_images(vr) for vr in vrs]
    
    # Process text and video inputs together
    model_inputs = processor(
        text=texts,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    # Special token IDs that represent image tokens in the model's vocabulary
    image_tokens = [151652, 151653, 151655, 151656]
    
    # Create labels for training by cloning input IDs
    labels = model_inputs["input_ids"].clone()

    # Mask system message and image token IDs in the labels
    # This prevents the model from being penalized for these tokens
    for i, _ in enumerate(labels):
        for image_token_id in image_tokens:
            labels[i][labels[i] == image_token_id] = -100
        labels[i][labels[i] == processor.tokenizer.pad_token_id] = -100

    model_inputs["labels"] = labels

    return model_inputs

def parse_args():
    """
    Parse command line arguments for training configuration.
    
    Returns:
        Namespace containing all parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train a Qwen2.5-VL model on cataract surgery videos")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model ID to use")
    parser.add_argument("--hf_token", type=str, help="HuggingFace token")
    parser.add_argument("--train_data_path", type=str, 
                        default="datasets/cataract1k/train_qa_pairs.json", 
                        help="Path to training data JSON")
    parser.add_argument("--train_video_dir", type=str, 
                        default="datasets/cataract1k/videos/train", 
                        help="Path to training video directory")
    parser.add_argument("--val_data_path", type=str, 
                        default="datasets/cataract1k/val_qa_pairs.json", 
                        help="Path to validation data JSON")
    parser.add_argument("--val_video_dir", type=str, 
                        default="datasets/cataract1k/videos/val", 
                        help="Path to validation video directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--push_to_hub", action="store_true", default=False, help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default="qwen2.5-vl-7b-instruct-cataract1k", help="Model ID for HuggingFace Hub")
    parser.add_argument("--use_qlora", action="store_true", default=True, help="Use QLoRA for training")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank parameter")
    parser.add_argument("--save_adapter", action="store_true", default=False, help="Save adapter locally")
    parser.add_argument("--save_dir", type=str, default="./qwen2.5-vl-7b-instruct-cataract1k", help="Directory to save adapter")
    return parser.parse_args()

def main():
    """
    Main function to set up and run the training process.
    """
    global processor  # Make processor globally accessible for the collate function
    args = parse_args()
    
    # Configure quantization if using QLoRA
    if args.use_qlora:
        # BitsAndBytes configuration for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            bnb_4bit_use_double_quant=True,  # Use double quantization for better precision
            bnb_4bit_quant_type="nf4",       # Use NF4 quantization type
            bnb_4bit_compute_type=torch.bfloat16  # Use bfloat16 for computation
        )
    else:
        bnb_config = None
    
    # Load the pre-trained model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, 
        quantization_config=bnb_config if args.use_qlora else None,
        device_map="auto",  # Automatically distribute model across available GPUs
    )
    
    # Load the processor for tokenization and image processing
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_id, padding_side="right", use_fast=True)
    
    # Configure LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        lora_alpha=args.lora_alpha,  # Scaling factor for LoRA
        lora_dropout=args.lora_dropout,  # Dropout probability for LoRA layers
        r=args.r,  # Rank of the LoRA matrices
        bias="none",  # Don't apply LoRA to bias terms
        target_modules=["q_proj", "v_proj"],  # Apply LoRA only to query and value projection matrices
        task_type="CAUSAL_LM",  # Task type for causal language modeling
    )
    
    # Apply LoRA configuration to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print the number and percentage of trainable parameters
    
    # Load training and validation datasets
    train_dataset = JSONLDataset(
        jsonl_file_path=args.train_data_path,
        video_directory_path=args.train_video_dir,
    )
    val_dataset = JSONLDataset(
        jsonl_file_path=args.val_data_path,
        video_directory_path=args.val_video_dir,
    )

    # Set environment variables for distributed training
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Configure training arguments using SFTConfig
    training_args = SFTConfig(
        output_dir=args.save_dir,  # Directory to save model checkpoints
        num_train_epochs=args.num_epochs,  # Number of training epochs
        per_device_train_batch_size=args.batch_size,  # Batch size per GPU for training
        per_device_eval_batch_size=args.batch_size,  # Batch size per GPU for evaluation
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Number of steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        optim="adamw_torch_fused",  # Use fused AdamW optimizer for better performance
        learning_rate=args.learning_rate,  # Learning rate
        lr_scheduler_type="cosine",  # Use cosine learning rate
        logging_steps=10,  # Log metrics every 10 steps
        eval_steps=10,  # Evaluate every 10 steps
        eval_strategy="steps",  # Evaluate based on steps, not epochs
        save_strategy="steps",  # Save based on steps, not epochs
        save_steps=100,  # Save checkpoint every 10 steps
        metric_for_best_model="eval_loss",  # Use evaluation loss to determine best model
        greater_is_better=False,  # Lower loss is better
        load_best_model_at_end=True,  # Load the best model at the end of training
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision on Ampere GPUs
        max_grad_norm=0.3,  # Clip gradients to prevent exploding gradients
        warmup_ratio=0.03,  # Warm up learning rate for 3% of training steps
        push_to_hub=args.push_to_hub,  # Whether to push model to HuggingFace Hub
        push_to_hub_token=args.hf_token,  # HuggingFace token for pushing to Hub
        push_to_hub_model_id=args.hub_model_id,  # Model ID for HuggingFace Hub
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Disable reentrant for gradient checkpointing
        dataset_kwargs={"skip_prepare_dataset": True},  # Skip dataset preparation in SFTTrainer
    )
    
    # Disable removal of unused columns to keep all data available
    training_args.remove_unused_columns = False
    
    # Initialize SFTTrainer for supervised fine-tuning
    trainer = SFTTrainer(
        model=model,  # The model to train
        args=training_args,  # Training arguments
        train_dataset=train_dataset,  # Training dataset
        data_collator=collate_fn,  # Custom collate function
        eval_dataset=val_dataset,  # Validation dataset
        peft_config=lora_config,  # LoRA configuration
        processing_class=processor.tokenizer,  # Tokenizer for processing text
    )
    
    # Start training
    trainer.train()
    
    # Save adapter locally if requested
    if args.save_adapter:
        trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main() 
