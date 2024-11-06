import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from unsloth import is_bfloat16_supported
import multiprocessing

# Parameters
max_seq_length = 512
dtype = torch.float16
load_in_4bit = True  # We will use 4-bit quantization

# Load model and tokenizer
model_name = "unsloth/Llama-3.2-3B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Prepare LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank of the LoRA matrices
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Attach LoRA adapters to the model
model = get_peft_model(model, lora_config)

# Preprocess text
def preprocess_text(text):
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    text = text[start_idx:end_idx]
    return text

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

cleaned_text = preprocess_text(text)
lines = cleaned_text.split('\n\n')
dataset = Dataset.from_dict({"text": lines})

# Tokenize dataset
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

num_cpus = multiprocessing.cpu_count()

dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=num_cpus,
    remove_columns=["text"],
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We are doing causal language modeling
)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=True,
    bf16=False,
    logging_steps=50,
    optim="paged_adamw_8bit",  # Use an optimizer compatible with 8-bit models
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    output_dir="outputs",
    report_to=["tensorboard"],
    logging_dir="logs",
)

# Trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    data_collator=data_collator,
    args=training_args,
)

# Train the model
trainer.train()

# Save the LoRA adapters (not the full model)
model.save_pretrained("shakespeare_model_lora")
tokenizer.save_pretrained("shakespeare_model_lora")
