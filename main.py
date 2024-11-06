from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from unsloth import is_bfloat16_supported
import multiprocessing

# Parameters
max_seq_length = 512
dtype = None
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

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
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=50,
    optim="adamw_torch",
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
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=data_collator,
    args=training_args,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("shakespeare_model")
tokenizer.save_pretrained("shakespeare_model")
