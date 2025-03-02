# Surgery-SFT: Cataract Surgery Video Analysis

This repository contains the code for my Applied Deep Learning (Winter Semester 24/25) Course Project at Ludwig Maximilian University of Munich. The project focuses on processing cataract surgery videos, generating question-answer pairs, and fine-tuning a vision-language model (Qwen2.5-VL) to analyze surgical procedures.

## Data Processing Pipeline

Follow these steps in order to prepare the dataset and train the model:

### 1. Cut Videos into Segments

```bash
python data_utils/cut_videos.py
```

This script processes the original cataract surgery videos by cutting them into smaller segments of length 2.5 seconds to 7.5 seconds for each surgical phase. It:
- Reads phase annotations from CSV files
- Cuts each video into 2.5 to 7.5 second intervals within each surgical phase
- Outputs the segmented videos to the `datasets/cataract1k/videos/` directory

### 2. Generate Object Information

```bash
python data_utils/object_generation.py
```

This script creates a JSON file mapping each video segment to its corresponding:
- Surgical phase
- Visible objects (instruments and anatomical structures)
- Timestamp information
- Video filename

The output is saved as `datasets/cataract1k/case_objects.json`, which serves as the foundation for generating question-answer pairs.

### 3. Generate Question-Answer Pairs

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

### 4. Split Dataset into Train/Validation Sets

```bash
python data_utils/split_dataset.py
```

This script:
- Splits the generated QA pairs into training (90%) and validation (10%) sets
- Uses a fixed random seed (42) for reproducibility
- Saves the splits as `train_qa_pairs.json` and `val_qa_pairs.json` in the `datasets/cataract1k/` directory

### 5. Organize Videos into Train/Test/Validation Folders

```bash
python data_utils/organize_videos.py
```

This script:
- Creates train, test, and validation directories
- Reads the train and validation QA pairs to identify which videos belong to each set
- Moves videos to their respective folders (train, test, val)
- Any videos not in the train or validation sets are moved to the test folder

### 6. Train the Model

```bash
python train.py
```

This script:
- Fine-tunes a Qwen2.5-VL model on the cataract surgery dataset
- Uses QLoRA (Quantized Low-Rank Adaptation) for efficient training
- Processes videos using the Decord library to extract frames
- Formats the data as a multi-turn conversation with system, user, and assistant messages
- Trains the model to answer questions about surgical phases, instruments, and anatomical structures
- Supports pushing the trained model to Hugging Face Hub

## Command Line Arguments

The training script supports various command line arguments:

```bash
python train.py --model_id "Qwen/Qwen2.5-VL-7B-Instruct" --batch_size 4 --num_epochs 1 --learning_rate 2e-4
```

For a full list of available arguments, run:

```bash
python train.py --help
```

## Dataset Structure

The dataset should be organized as follows:

```bash
datasets/
└── cataract1k/
├── annotations/
│ ├── phase_annotations/
│ └── segment_annotations/
├── videos/
│ ├── train/
│ ├── test/
│ └── val/
├── case_objects.json
├── qa_pairs_without_idle.json
├── train_qa_pairs.json
└── val_qa_pairs.json
```


## Inference

After training, you can use the model for inference by running the provided `run_inference.ipynb` notebook.

