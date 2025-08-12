import torch
import tiktoken
import argparse

from model import GPTSpamClassification
from utils import GPT2_CONFIG_124M
from dataloader import create_spam_dataloader

def classify_text(text, model, tokenizer, device, max_length):
    """Classifies a single piece of text as spam or ham."""
    model.eval()
    
    # Prepare the input
    token_ids = tokenizer.encode(text)
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]  # Truncate if too long
    
    # Pad the sequence
    pad_token_id = 50256
    padded_tokens = torch.full((max_length,), pad_token_id, dtype=torch.long)
    padded_tokens[:len(token_ids)] = torch.tensor(token_ids)
    
    input_tensor = padded_tokens.unsqueeze(0).to(device) # Add batch dimension
    
    # Get the model's prediction
    with torch.no_grad():
        logits = model(input_tensor)
    
    predicted_label = torch.argmax(logits[:, -1, :], dim=-1).item()
    
    return "Spam" if predicted_label == 1 else "Ham"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if a text is spam or ham.")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="The text message to classify."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/finetuned_spam.pth",
        help="Path to the fine-tuned model checkpoint."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    try:
        train_loader = create_spam_dataloader("data/train.csv", shuffle=False)
        max_length = train_loader.dataset.max_length
    except FileNotFoundError:
        print("Error: train.csv not found. Please run prepare_data.py first.")
        exit()
    # Load the fine-tuned model
    model = GPTSpamClassification(GPT2_CONFIG_124M).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model_path}'.")
        print("Please run `python train.py --classification` to fine-tune the model first.")
        exit()
        
    # Classify the input text
    prediction = classify_text(args.text, model, tokenizer, device, max_length)

    print(f"\nPrompt: '{args.text}'")
    print(f"\nPrediction: {prediction}")