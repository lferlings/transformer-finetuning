from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, DefaultDataCollator
from unsloth import is_bfloat16_supported
import multiprocessing

max_seq_length = 4096
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        )
        for convo in convos
    ]
    return {"text": texts}

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched=True)

# Preprocess the dataset
def preprocess_function(examples):
    inputs = examples["text"]
    model_inputs = tokenizer(
        inputs,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    )
    return model_inputs

num_cpus = multiprocessing.cpu_count()
dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_cpus,
    remove_columns=dataset.column_names,
)

data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,  # Adjust as needed
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to=["tensorboard"],  # Enable TensorBoard logging
    logging_dir="/home/jovyan/work/saved_data/runs",         # Specify the logging directory
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=data_collator,
    args=training_args,
)

trainer_stats = trainer.train()

model.save_pretrained("/home/jovyan/work/saved_data/transformer-finetuning/models/iteration_1/lora_model")  # Local saving
tokenizer.save_pretrained("/home/jovyan/work/saved_data/transformer-finetuning/models/iteration_1/lora_model")
