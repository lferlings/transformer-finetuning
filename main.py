import os
import random
import string
import torch
from torch.optim.adamw import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Train GPT-2 on a text dataset.")
parser.add_argument('--model', type=str, required=True, help="Path to the pretrained model or model name (e.g., 'gpt2').")
parser.add_argument('--batch_size', type=int, required=False, default=4, help="Batch size for training.")
parser.add_argument('--epochs', type=int, required=True, help="Number of training epochs.")
parser.add_argument('--output_dir', type=str, required=False,default='./results/', help="Number of training epochs.")

# Parse arguments
args = parser.parse_args()

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Load text file as dataset
with open('./scraper/songs.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Initialize GPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(args.model)
model = GPT2LMHeadModel.from_pretrained(args.model)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize text and convert to torch tensors
lyrics_list = text.split('----------------------------------------\n')
processed_strings = ["\n".join(item.split("\n")[1:]) for item in lyrics_list]

# Ensure tokenizer has a pad token
tokenizer.pad_token = tokenizer.eos_token
encodings = tokenizer(
    processed_strings,
    return_tensors='pt',
    max_length=512,
    truncation=True,
    padding=True  # Ensures all inputs are padded to the same length
)

# Move input IDs to the correct device (CPU/GPU)
input_ids = encodings['input_ids'].to(device)
attention_mask = encodings['attention_mask'].to(device)

# Set training parameters
train_batch_size = args.batch_size
num_train_epochs = args.epochs
learning_rate = 5e-5

# Initialize optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(input_ids) * num_train_epochs // train_batch_size
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Train the model
model.train()
for epoch in range(num_train_epochs):
    epoch_loss = 0.0
    for i in range(0, len(input_ids), train_batch_size):
        # Slice the input IDs and attention masks for the current batch
        batch_input_ids = input_ids[i:i+train_batch_size]
        batch_attention_mask = attention_mask[i:i+train_batch_size]
        # Create labels from input_ids
        batch_labels = batch_input_ids.clone()
        batch_labels[batch_labels == tokenizer.pad_token_id] = -100  # Ignore pad tokens in the loss
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss
        # Backward pass
        loss.backward()
        epoch_loss += loss.item()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    print('Epoch: {}, Loss: {:.4f}'.format(epoch + 1, epoch_loss / len(input_ids)))

# Save the trained model

