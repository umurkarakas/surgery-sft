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
decord.bridge.set_bridge('torch')

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, video_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.video_directory_path = video_directory_path
        self.entries = self._load_entries()

    def _load_entries(self):
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            data = json.load(file)
            for entry_list in data:
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
        return len(self.entries)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        video_path = os.path.join(self.video_directory_path, entry['video_filename'])
        
        # Load video using decord
        vr = decord.VideoReader(video_path)
        
        return vr, entry, format_data(self.video_directory_path, entry)

def init_distributed():
    """Initialize distributed environment"""
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        if not dist.is_initialized():
            # Initialize the distributed environment
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
            print(f"Initialized distributed environment: rank {dist.get_rank()} of {dist.get_world_size()}")
    else:
        print("Running in non-distributed mode")

def format_data(video_directory_path, entry):
    SYSTEM_MESSAGE = """You are a vision language model specialized in analyzing cataract surgery videos.
Your task is to analyze the provided surgical video frames and extract relevant clinical information from:
- Current surgical phase/step
- Visible surgical instruments in use
- Visible and relevant anatomical structures

Focus on providing accurate, clinically relevant observations using proper medical terminology.
Base your responses solely on what is visible in the video frames.
Provide clear, concise answers."""

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

def sample_images(vr, sample_fps=8):
    num_frames = vr._num_frame
    frames_idx = [int(vr.get_avg_fps() / sample_fps)*i for i in range(num_frames // int(vr.get_avg_fps()))] 
    return vr.get_batch(frames_idx)

def collate_fn(batch):
    vrs, _, examples = zip(*batch)

    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    video_inputs = [sample_images(vr) for vr in vrs]
    
    model_inputs = processor(
        text=texts,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    image_tokens = [151652, 151653, 151655, 151656]
    
    labels = model_inputs["input_ids"].clone()

    # mask system message and image token IDs in the labels
    for i, _ in enumerate(labels):
        for image_token_id in image_tokens:
            labels[i][labels[i] == image_token_id] = -100
        labels[i][labels[i] == processor.tokenizer.pad_token_id] = -100

    model_inputs["labels"] = labels

    return model_inputs

def parse_args():
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
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--push_to_hub", action="store_true", default=False, help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default="qwen2.5-7b-instruct-cataract1k", help="Model ID for HuggingFace Hub")
    parser.add_argument("--use_qlora", action="store_true", default=True, help="Use QLoRA for training")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank parameter")
    parser.add_argument("--save_adapter", action="store_true", default=False, help="Save adapter locally")
    parser.add_argument("--save_dir", type=str, default="./qwen2.5-7b-instruct-cataract1k", help="Directory to save adapter")
    return parser.parse_args()

def main():
    global processor
    args = parse_args()
    
    # Configure quantization if using QLoRA
    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16
        )
    else:
        bnb_config = None
    
    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, 
        quantization_config=bnb_config if args.use_qlora else None,
        device_map="auto",
    )
    
    # Load processor
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_id, padding_side="right", use_fast=True)
    
    # Configure LoRA
    lora_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.r,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    train_dataset = JSONLDataset(
        jsonl_file_path=args.train_data_path,
        video_directory_path=args.train_video_dir,
    )
    val_dataset = JSONLDataset(
        jsonl_file_path=args.val_data_path,
        video_directory_path=args.val_video_dir,
    )

    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    # Configure training arguments
    training_args = SFTConfig(
        output_dir=args.save_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        logging_steps=10,
        eval_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=10,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=args.push_to_hub,
        push_to_hub_token=args.hf_token,
        push_to_hub_model_id=args.hub_model_id,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    
    training_args.remove_unused_columns = False
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=processor.tokenizer,
    )
    
    # Train model
    trainer.train()
    
    # Save adapter locally if requested
    if args.save_adapter:
        trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main() 
