from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


# TinyLlama/TinyLlama_v1.1 - https://huggingface.co/TinyLlama/TinyLlama_v1.1
model_name = "TinyLlama/TinyLlama_v1.1"
print("-- Model and Tokenizer loading up!")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": "cpu"})

# # LoRA configs
print("-- Applying LoRA configs")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)

# Dataset
print("-- Loading dataset")
dataset = load_dataset("json", data_files="./data/dataset.json")["train"]

def format_and_tokenize(sample):
    input_data = sample["messages"][0]["content"]
    response_data = sample["messages"][1]["content"]

    text = f"Instruction: {input_data}\nInput: \nResponse: {response_data}"
    tokens = tokenizer(text, truncate=True, padding="max_length", max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

# Training config
training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    logging_dir="./logs"
)

# Train!
print("-- Starting to train!")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()