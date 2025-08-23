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


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Interactive chat with a GPT-2 model.")
    parser.add_argument(
        "--openai_gpt2",
        action="store_true",
        default=False,
        help="Use the official GPT-2 pretrained weights."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/trained_model_model_only.pth",
        help="Path to the trained model checkpoint."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.4,
        help="Temperature for sampling. Higher is more creative."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=25,
        help="Top-k filtering. Narrows the choices for the next token."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate per turn."
    )
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)

    # --- Load Model and Tokenizer ---
    if args.openai_gpt2:
        print("Loading OpenAI pre-trained GPT-2 weights...")
        model = GPTModel(GPT2_CONFIG_124M).to(device)
        model = load_weights_from_gpt2(model)
    else:
        try:
            model = GPTModel(Custom_GPT2_config).to(device)
            model.load_state_dict(torch.load(
                args.model_path, map_location=device))
        except FileNotFoundError:
            print(
                f"Error: Model file not found at '{args.model_path}'. Please run train.py first.")
            exit()
 
    model.eval()  # Set model to evaluation mode
    tokenizer = tiktoken.get_encoding("gpt2")
    print("Model loaded successfully. Type 'quit' or 'exit' to end the chat.")

    # --- Interactive Chat Loop ---
    while True:
        print("-" * 60)
        try:
            prompt = input("You: ")
            if prompt.lower() in ["quit", "exit"]:
                print("Bot: Goodbye!")
                break

            encoded_prompt = text_to_token_ids(prompt, tokenizer).to(device)

            with torch.no_grad():
                output_ids = generate(
                    model=model,
                    input_ids=encoded_prompt,
                    max_new_tokens=args.max_tokens,
                    context_size=GPT2_CONFIG_124M["context_length"] if args.openai_gpt2 else Custom_GPT2_config["context_length"],
                    temperature=args.temperature,
                    top_k=args.top_k
                )

                result = token_ids_to_text(output_ids, tokenizer)
            print(f"Bot: {result.strip()}")

        except KeyboardInterrupt:
            print("\nBot: Goodbye!")
            break

if __name__ == "__main__":
    main()
