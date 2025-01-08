import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(model_path, prompt, max_length=50):
    # Load the tokenizer and model from the specified directory
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    
    # Set device to GPU if available
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text using the model
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=False,       
            top_k=0,               
            top_p=1.0,             
            temperature=1.0,       
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated tokens to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate text using a model.")
    parser.add_argument('--model', type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument('--prompt', type=str, required=True, help="Input prompt for text generation.")
    parser.add_argument('--max_length', type=int, default=512, help="Maximum length of generated text.")
    parser.add_argument('--gpu', type=int, default=0, help="cuda:x to use as device.")
    args = parser.parse_args()
    
    # Generate text
    output_text = generate_text(args.model, args.prompt, args.max_length)
    print("Generated Text:\n")
    print(output_text)
