from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Load the raw text file
file_path = 'scraper/songs.txt'  # Replace with the actual path to your text file
with open(file_path, 'r', encoding='utf-8') as file:
    raw_text = file.read()

# Split the raw text by new lines
chunks = raw_text.split('\n')

# Remove empty lines (optional, if needed)
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

# Convert the list of chunks into a format suitable for Hugging Face's Dataset
dataset = Dataset.from_dict({'text': chunks})

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'  # You can choose another base model like 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad token to be the eos token
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding

# Tokenize the text data with max_length and padding to max_length
def tokenize_function(examples):
    # Tokenize the text
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    
    # For language modeling, the labels are the same as the input IDs but shifted by one position
    tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()  # Copy the input IDs for the labels
    return tokenized_inputs

dataset = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    overwrite_output_dir=True,      # Whether to overwrite the content of the output directory
    num_train_epochs=3,             # Number of training epochs
    per_device_train_batch_size=4,  # Batch size for training (adjust as needed)
    logging_dir='./logs',           # Directory for storing logs
    logging_steps=10,               # Log every 10 steps
)

# Set up the trainer
trainer = Trainer(
    model=model,                         # Pre-trained model
    args=training_args,                  # Training arguments
    train_dataset=dataset,               # Training dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('./moneyboy_model')

# Optionally, you can test the fine-tuned model
# Load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained('./moneyboy_model')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate text with a given prompt
input_text = "Yo, ich bin der"
inputs = tokenizer(input_text, return_tensors="pt", padding=True)

# Ensure input tensor has padding, as needed
outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)

# Decode the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
