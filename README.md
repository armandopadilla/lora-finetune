# TinyLlama FIFA World Cup Fine-Tuning

Fine-tune [TinyLlama v1.1](https://huggingface.co/TinyLlama/TinyLlama_v1.1) with LoRA (Low-Rank Adaptation) on a custom question-answer dataset about FIFA World Cup history. The project uses Hugging Face Transformers, PEFT, and Datasets to train a lightweight adapter and run inference with the fine-tuned model.

## Important

Yes, i know RAG would have been a better choice in this instance. I just wanted to use LoRA on something. To learn the different steps. 

## Overview

This repository demonstrates parameter-efficient fine-tuning of a small language model for domain-specific Q&A. Instead of updating the full model weights, it trains a LoRA adapter on top of the frozen base model, keeping memory requirements low and training fast enough to run on CPU.

The included dataset contains ~430 instruction-response pairs covering World Cup history from 1930 onward.

## Project Structure

```
fine_tunning/
├── data/
│   └── dataset.json          # Q&A training data (chat-style messages)
├── src/
│   ├── main.py               # Training script
│   └── test.py               # Inference script
├── requirements.txt          # Python dependencies
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch
- Hugging Face libraries: `transformers`, `peft`, `datasets`

Tested with:

| Package       | Version |
|---------------|---------|
| Python        | 3.13    |
| torch         | 2.12.0  |
| transformers  | 5.11.0  |
| peft          | 0.19.1  |
| datasets      | 5.0.0   |
| accelerate    | 1.14.0  |
| sentencepiece | 0.2.1   |

## Setup

1. Clone the repository:

```bash
git clone <your-repo-url>
cd fine_tunning
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** If `torch` fails to install for your platform, install it first from [pytorch.org](https://pytorch.org/get-started/locally/) for the correct CPU or CUDA build, then run `pip install -r requirements.txt` again.

## Training

Run the training script from the project root:

```bash
python src/main.py
```

This will:

1. Load `TinyLlama/TinyLlama_v1.1` from Hugging Face
2. Apply a LoRA adapter to the attention projection layers (`q_proj`, `v_proj`)
3. Tokenize and format the dataset from `data/dataset.json`
4. Train for 4 epochs and save checkpoints to `tinyllama-finetuned/`

### Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Epochs | 4 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | 5e-4 |
| Max sequence length | 256 |
| Device | CPU |

Checkpoints are saved at the end of each epoch (`checkpoint-54`, `checkpoint-108`, `checkpoint-162`, `checkpoint-216` for a full 4-epoch run).

Training logs are written to `./logs`.

## Inference

After training, run the test script to generate a response with the latest checkpoint:

```bash
python src/test.py
```

By default, `test.py` loads the adapter from `./tinyllama-finetuned/checkpoint-216` and prompts the model with:

> Was there any World Cup final that ended because of a fire?

To use a different checkpoint or prompt, edit the `adapter_path` and `prompt` variables in `src/test.py`.

## Dataset Format

Training data lives in `data/dataset.json` as a JSON array. Each entry follows a chat-style message format:

```json
{
  "messages": [
    {"role": "user", "content": "Where and when was the 1930 FIFA World Cup held?"},
    {"role": "assistant", "content": "The 1930 FIFA World Cup took place in Uruguay from 13 to 30 July 1930..."}
  ]
}
```

During training, each sample is formatted as:

```
Instruction: {user message}
Input: 
Response: {assistant message}
```

To fine-tune on your own data, replace or extend `data/dataset.json` using the same structure.

## Model Details

- **Base model:** [TinyLlama/TinyLlama_v1.1](https://huggingface.co/TinyLlama/TinyLlama_v1.1) (1.1B parameters)
- **Fine-tuning method:** LoRA via [PEFT](https://github.com/huggingface/peft)
- **Output:** LoRA adapter weights saved under `tinyllama-finetuned/`

The base model weights are downloaded from Hugging Face on first run and are not stored in this repository.

## Tips

- **GPU training:** Change `device_map={"": "cpu"}` in `src/main.py` to use a CUDA or MPS device for faster training.
- **Checkpoint selection:** Later checkpoints correspond to more training; use the final checkpoint unless you observe overfitting on a validation set.
- **Large artifacts:** Consider adding `tinyllama-finetuned/`, `.venv/`, and `logs/` to `.gitignore` if you do not want to commit model checkpoints or local environment files.

## License

This project fine-tunes TinyLlama, which is released under the Apache 2.0 license. See the [model card](https://huggingface.co/TinyLlama/TinyLlama_v1.1) for details.
