import torch
import tiktoken
import argparse
import os
from dataloader import create_spam_dataloader
from model import GPTSpamClassification
from utils import GPT2_CONFIG_124M


def classify_text(text, model, tokenizer, device, max_length):
    """Classifies a single piece of text as spam or ham."""
    model.eval()

    # Prepare the input
    token_ids = tokenizer.encode(text)
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]  # Truncate if too long

    # Use padding
    pad_token_id = tokenizer.encode(
        "<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    padded_tokens = torch.full((max_length,), pad_token_id, dtype=torch.long)
    padded_tokens[:len(token_ids)] = torch.tensor(token_ids)

    input_tensor = padded_tokens.unsqueeze(0).to(device)  # Add batch dimension

    # Get the model's prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits[:, -1, :], dim=-1)

    predicted_label = torch.argmax(logits[:, -1, :], dim=-1).item()
    confidence = probs[0, predicted_label].item()

    classification = "Spam" if predicted_label == 1 else "Ham"
    return classification, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Interactive spam/ham classifier using GPT-2."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/finetuned_spam_model_only.pth",
        help="Path to the fine-tuned model checkpoint."
    )
    parser.add_argument(
        "--show_confidence",
        action="store_true",
        help="Show confidence scores with predictions."
    )
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)

    print(f"Using device: {device}")

    # --- Load Model and Tokenizer ---
    tokenizer = tiktoken.get_encoding("gpt2")
    try:
        # Load the training dataloader to determine the max_length the model was trained on
        train_loader = create_spam_dataloader("data/train.csv", shuffle=False)
        actual_max_length = train_loader.dataset.max_length
    except FileNotFoundError:
        print("Error: train.csv not found. Please run prepare_data.py first.")
        exit()

    # Load the fine-tuned model
    model = GPTSpamClassification(GPT2_CONFIG_124M).to(device)

    try:
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(
                args.model_path, map_location=device))
            print(f"Model loaded successfully from {args.model_path}")
        else:
            print(f"Error: Model file not found at '{args.model_path}'.")
            print(
                "Please run `python fine_tune.py --classification_ft` to fine-tune the model first.")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    print("\n" + "="*60)
    print("🛡️  SPAM/HAM CLASSIFIER READY")
    print("="*60)
    print("Enter messages to classify as spam or ham.")
    print("Commands:")
    print("  - Type 'quit' or 'exit' to end the session")
    print("  - Type 'help' for usage tips")
    print("  - Type 'stats' to see model information")
    if args.show_confidence:
        print("  - Confidence scores are enabled")
    print("-" * 60)

    # --- Interactive Classification Loop ---
    message_count = 0

    while True:
        try:
            print(f"\n[Message #{message_count + 1}]")
            user_input = input("Enter message: ").strip()
            # print(user_input)
            if not user_input:
                print("⚠️  Please enter a message to classify.")
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("👋 Goodbye! Thanks for using the spam classifier.")
                break

            elif user_input.lower() == "help":
                print("\n📋 HELP:")
                print("• Enter any text message to classify it as spam or ham")
                print("• Spam: Unwanted/promotional messages")
                print("• Ham: Normal/legitimate messages")
                print("• Examples:")
                print("  - 'Congratulations! You've won lottery of $1000! Send me your bank details.' → Likely Spam")
                print("  - 'Hey, are we still meeting for lunch?' → Likely Ham")
                continue

            elif user_input.lower() == "stats":
                print(f"\n📊 MODEL STATISTICS:")
                print(f"• Model path: {args.model_path}")
                print(f"• Device: {device}")
                print(f"• Messages classified: {message_count}")

                # Count model parameters
                total_params = sum(p.numel() for p in model.parameters())
                print(f"• Model parameters: {total_params:,} total")
                continue

            # Classify the input
            classification, confidence = classify_text(
                user_input, model, tokenizer, device, actual_max_length
            )

            message_count += 1

            # Display result with appropriate emoji
            emoji = "🚫" if classification == "Spam" else "✅"
            print(f"\n{emoji} Classification: {classification}")

            if args.show_confidence:
                print(f"🎯 Confidence: {confidence:.1%}")

                # Add confidence interpretation
                if confidence >= 0.9:
                    confidence_level = "Very High"
                elif confidence >= 0.7:
                    confidence_level = "High"
                elif confidence >= 0.6:
                    confidence_level = "Moderate"
                else:
                    confidence_level = "Low"
                print(f"📊 Confidence Level: {confidence_level}")

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during classification: {e}")
            continue

    print(f"\n📈 Session Summary: Classified {message_count} messages")


if __name__ == "__main__":
    main()
