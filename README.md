# Surgery-SFT: Cataract Surgery Video Analysis

This repository contains the code for my Applied Deep Learning (Winter Semester 24/25) Course Project at Ludwig Maximilian University of Munich. The project focuses on processing cataract surgery videos, generating question-answer pairs, and fine-tuning a vision-language model (Qwen2.5-VL) to analyze surgical procedures.

## Trained Models

The following fine-tuned models are available on the Hugging Face Hub:

- [Qwen2.5-VL-7B-Instruct-Cataract1K](https://huggingface.co/kida1122/qwen2.5-vl-7b-instruct-cataract1k): Fine-tuned Qwen2.5-VL model for cataract surgery video analysis
- [LLaVA-NeXT-Video-7B-Cataract1K](https://huggingface.co/kida1122/llava-next-video-7b-cataract1k): Fine-tuned LLaVA-NeXT-Video model for cataract surgery video analysis

## Data Processing Pipeline

Follow these steps in order to prepare the dataset and train the model:

### 1. Download and Install the Dataset

First, download the Cataract-1K dataset from Dropbox:
- [Download Cataract-1K Dataset (reshaped into 224x224 videos)](https://www.dropbox.com/scl/fi/5ybj7gd07hd38x1pezwdr/surgery-Cataract-1K.zip?rlkey=42wja3aptub866l487k2dhfmm&dl=0)

Extract the downloaded zip file in the base directory of this repository. The phase and segment annotations are retrieved with cloning the repository. The initial structure of the datasets should look like:

```bash
surgery-sft/
├── datasets/
│   └── cataract1k/
│       ├── annotations/
│       │   ├── phase_annotations/
│       │   └── segment_annotations/
│       └── videos/
└── surgery-Cataract-1K/
    └── Phase_recognition_dataset/
        └── videos_224/
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Cut Videos into Segments

```bash
python data_utils/cut_videos.py
```

This script processes the original cataract surgery videos by cutting them into smaller segments of length 2.5 seconds to 7.5 seconds for each surgical phase. It:
- Reads phase annotations from CSV files
- Cuts each video into 2.5 to 7.5 second intervals within each surgical phase
- Outputs the segmented videos to the `datasets/cataract1k/videos/` directory

### 4. Generate Object Information

```bash
python data_utils/object_generation.py
```

This script creates a JSON file mapping each video segment to its corresponding:
- Surgical phase
- Visible objects (instruments and anatomical structures)
- Timestamp information
- Video filename

The output is saved as `datasets/cataract1k/case_objects.json`, which serves as the foundation for generating question-answer pairs.

### 5. Generate Question-Answer Pairs

```bash
python data_utils/qa_generation.py
```

This script:
- Uses a quantized large language model (Mistral-Small-24B) to generate question-answer pairs for each video segment
- Creates three QA pairs for each segment:
  1. Identifying the current surgical phase
  2. Describing visible anatomical structures
  3. Identifying visible surgical instruments
- Formats the output as structured JSON
- Saves the results to `datasets/cataract1k/qa_pairs_without_idle.json`

### 6. Split Dataset into Train/Validation Sets

```bash
python data_utils/split_dataset.py
```

This script:
- Splits the generated QA pairs into training (90%) and validation (10%) sets
- Uses a fixed random seed (42) for reproducibility
- Saves the splits as `train_qa_pairs.json` and `val_qa_pairs.json` in the `datasets/cataract1k/` directory

### 7. Organize Videos into Train/Test/Validation Folders

```bash
python data_utils/organize_videos.py
```

This script:
- Creates train, test, and validation directories
- Reads the train and validation QA pairs to identify which videos belong to each set
- Moves videos to their respective folders (train, test, val)
- Any videos that do not have segment labels (which are also not in train and validation sets) are moved to the test folder

## Final Dataset Structure

The dataset should be organized as follows before starting to finetune the model:

```bash
surgery-sft/
├── datasets/
│   └── cataract1k/
│       ├── annotations/
│       │   ├── phase_annotations/
│       │   └── segment_annotations/
│       ├── videos/
│       │   ├── train/
│       │   ├── test/
│       │   └── val/
│       ├── case_objects.json
│       ├── qa_pairs_without_idle.json
│       ├── train_qa_pairs.json
│       └── val_qa_pairs.json
└── surgery-Cataract-1K/  # Original dataset directory
```

## Fine-tune the Model

The repository provides two training scripts for fine-tuning different vision-language models on the cataract surgery dataset:

### 1. Fine-tune Qwen2.5-VL

```bash
python train.py
```

This script fine-tunes the Qwen2.5-VL model using the dependencies specified in `requirements.txt`.

### 2. Fine-tune LLaVA-NeXT-Video

```bash
# First install the required version of transformers
pip install transformers==4.48.0

# Then run the training script
python train_llava.py
```

Note: LLaVA-NeXT-Video specifically requires `transformers==4.48.0`, which differs from the version used for Qwen2.5-VL. Make sure to install this specific version before running the LLaVA training script.

### Common Features of Both Training Scripts

Both scripts:
- Use QLoRA (Quantized Low-Rank Adaptation) for efficient training
- Process videos using the Decord library to extract frames
- Format the data as a multi-turn conversation with system, user, and assistant messages
- Train the model to answer questions about surgical phases, instruments, and anatomical structures
- Support pushing the trained model to Hugging Face Hub

## Command Line Arguments

The training scripts support various command line arguments to customize the training process:

```bash
# For Qwen2.5-VL
python train.py --model_id "Qwen/Qwen2.5-VL-7B-Instruct" --batch_size 4 --num_epochs 1 --learning_rate 2e-5

# For LLaVA-NeXT-Video
python train_llava.py --model_id "llava-hf/LLaVa-NeXT-Video-7b-hf" --batch_size 1 --num_epochs 1 --learning_rate 2e-5
```

### Available Arguments:

| Argument | Default (Qwen) | Default (LLaVA) | Description |
|----------|----------------|-----------------|-------------|
| `--model_id` | "Qwen/Qwen2.5-VL-7B-Instruct" | "llava-hf/LLaVa-NeXT-Video-7b-hf" | Base model ID to use from Hugging Face |
| `--hf_token` | None | None | HuggingFace token for accessing gated models and pushing to Hub |
| `--train_data_path` | "datasets/cataract1k/train_qa_pairs.json" | "datasets/cataract1k/train_qa_pairs.json" | Path to training data JSON file |
| `--train_video_dir` | "datasets/cataract1k/videos/train" | "datasets/cataract1k/videos/train" | Path to training video directory |
| `--val_data_path` | "datasets/cataract1k/val_qa_pairs.json" | "datasets/cataract1k/val_qa_pairs.json" | Path to validation data JSON file |
| `--val_video_dir` | "datasets/cataract1k/videos/val" | "datasets/cataract1k/videos/val" | Path to validation video directory |
| `--batch_size` | 4 | 1 | Batch size for training and evaluation |
| `--num_epochs` | 2 | 1 | Number of training epochs |
| `--learning_rate` | 2e-5 | 2e-5 | Learning rate for optimization |
| `--gradient_accumulation_steps` | 32 | 32 | Number of steps to accumulate gradients before updating weights |
| `--push_to_hub` | False | False | Whether to push the model to HuggingFace Hub after training |
| `--hub_model_id` | "qwen2.5-vl-7b-instruct-cataract1k" | "llava-next-video-7b-cataract1k" | Model ID for HuggingFace Hub when pushing |
| `--use_qlora` | True | True | Whether to use QLoRA for efficient fine-tuning |
| `--lora_alpha` | 32 | 32 | LoRA alpha parameter (scaling factor) |
| `--lora_dropout` | 0.05 | 0.1 | Dropout probability for LoRA layers |
| `--r` | 32 | 32 | LoRA rank parameter (lower means fewer parameters) |
| `--save_adapter` | False | False | Whether to save the adapter locally |
| `--save_dir` | "./qwen2.5-vl-7b-instruct-cataract1k" | "./llava-next-video-7b-cataract1k" | Directory to save the adapter if `save_adapter` is True |

For a full list of available arguments, run:

```bash
python train.py --help
# or
python train_llava.py --help
```


## Inference

After training, you can use the model for inference by running the provided `run_inference.ipynb` notebook.

