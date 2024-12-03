import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.adamw import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Train Llama on a text dataset.")
parser.add_argument('--model', type=str, required=True, help="Path to the pretrained model or model name (e.g., 'meta-llama/Llama-3.2-1B').")
parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training.")
parser.add_argument('--epochs', type=int, required=True, help="Number of training epochs.")
parser.add_argument('--output_dir', type=str, default='./outputs/llama/', help="Path to save the model.")

# Parse arguments
args = parser.parse_args()

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

def print_gpu_memory(step_desc):
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(f"{step_desc} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


# Load text file as dataset
with open('./scraper/songs.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create a custom Dataset class
class LyricsDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'  # Ensures consistent input size
        )
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore pad tokens in the loss
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Prepare the dataset and dataloader
lyrics_list = text.split('----------------------------------------\n')
processed_strings = ["\n".join(item.split("\n")[1:]) for item in lyrics_list]

dataset = LyricsDataset(processed_strings, tokenizer)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Set training parameters
num_train_epochs = args.epochs
learning_rate = 5e-5

# Initialize optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_train_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Use mixed precision training if supported
scaler = torch.amp.GradScaler()

# Training loop
model.train()
for epoch in range(num_train_epochs):
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch_input_ids = batch['input_ids'].to(device, non_blocking=True)
        batch_attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        batch_labels = batch['labels'].to(device, non_blocking=True)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels
            )
            loss = outputs.loss
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Accumulate loss
        epoch_loss += loss.item()
        
        # Delete variables to free up memory
        del batch_input_ids, batch_attention_mask, batch_labels, outputs, loss
        torch.cuda.empty_cache()
        
        if batch_idx % 10 == 0:
            print_gpu_memory(f"After batch {batch_idx}")
    
    # Calculate average loss
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f'Epoch: {epoch + 1}, Loss: {avg_epoch_loss:.4f}')
    
    # Save the model at the end of each epoch
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
