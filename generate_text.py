import torch
import tiktoken
import argparse
from model import GPTModel
from utils import (
    generate,
    GPT2_CONFIG_124M,
    Custom_GPT2_config,
    text_to_token_ids,
    token_ids_to_text,
    load_weights_from_gpt2
)


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Generate text using a trained GPT model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/trained_model.pth",
        help="Path to the trained model checkpoint."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Every effort moves you",
        help="The starting prompt for text generation."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.4,
        help="Temperature for sampling (0.0 for greedy)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=25,
        help="Top-k filtering for sampling."
    )
    parser.add_argument(
        "--from_pretrained",
        action="store_true",
        default=False,
        help="Use the official GPT-2 pretrained weights."
    )
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)

    # --- Model Loading ---
    if args.from_pretrained:
        print("Loading pre-trained GPT-2 weights...")
        model = GPTModel(GPT2_CONFIG_124M).to(device)
        model = load_weights_from_gpt2(model)
    else:
        try:
            model = GPTModel(Custom_GPT2_config).to(device)
            model.load_state_dict(torch.load(
                args.model_path, map_location=device))
        except FileNotFoundError:
            print(f"Error: Model file not found at '{args.model_path}'. Please run train.py first.")
            exit()

    model.eval()  # Set model to evaluation mode
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Generating text from prompt: '{args.prompt}'")
    encoded_prompt = text_to_token_ids(args.prompt, tokenizer).to(device)

    with torch.no_grad():
        output_ids = generate(
            model=model,
            input_ids=encoded_prompt,
            max_new_tokens=args.max_tokens,
            context_size=GPT2_CONFIG_124M["context_length"] if args.from_pretrained else Custom_GPT2_config["context_length"],  
            temperature=args.temperature,
            top_k=args.top_k
        )

    result = token_ids_to_text(output_ids, tokenizer)
    print("\n--- Generated Text ---")
    print(result)