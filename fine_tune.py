import argparse
import tiktoken
import torch
import os
import json
import time
from utils import (
    GPT2_CONFIG_124M,
    load_weights_from_gpt2
)
from train import train_model, plot_losses
from model import GPTModel, GPTSpamClassification
from dataloader import create_spam_dataloader, create_instruction_dataloader, format_input
import matplotlib.pyplot as plt


def evaluate_classifier(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    train_loss, val_loss = 0., 0.
    with torch.no_grad():
        # Calculate loss for the training set
        for i, (input_batch, target_batch) in enumerate(train_loader):
            if i >= eval_iter:
                break
            input_batch, target_batch = input_batch.to(
                device), target_batch.to(device)
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(
                logits[:, -1], target_batch)
            train_loss += loss.item()

        # Calculate loss for the validation set
        for i, (input_batch, target_batch) in enumerate(val_loader):
            if i >= eval_iter:
                break
            input_batch, target_batch = input_batch.to(
                device), target_batch.to(device)
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(
                logits[:, -1], target_batch)
            val_loss += loss.item()

    model.train()
    if eval_iter == 0:
        return 0., 0.
    return train_loss / eval_iter, val_loss / eval_iter


def plot_values(epochs_seen, examples_seen, train_values, val_values, save_path, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.",
             label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"{label.capitalize()} plot saved to {save_path}")


def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_epoch=0):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for input_batch, target_labels in train_loader:
            optimizer.zero_grad()
            input_batch, target_labels = input_batch.to(
                device), target_labels.to(device)
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(
                logits[:, -1], target_labels)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_classifier(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        # Evaluation loop
        model.eval()
        train_acc = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_acc = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter)
        print(
            f"Epoch {epoch+1}: Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}%"
        )
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    return train_losses, val_losses, train_accs, val_accs, examples_seen


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(
                device), target_batch.to(device)

            with torch.no_grad():
                # Logits of last output token
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels ==
                                    target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def evaluate_instruction_model(model, data_loader, device, tokenizer, num_examples=5):
    """Evaluate instruction model by generating responses to a few examples."""
    model.eval()
    print("\n--- Sample Instruction Evaluations ---")

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_examples:
                break

            input_ids = input_batch[0].to(device)
            input_text = tokenizer.decode(input_ids.cpu().tolist())
            if "### Response:" in input_text:
                instruction_part = input_text.split("### Response:")[
                    0] + "### Response:"
                instruction_ids = tokenizer.encode(instruction_part)
                instruction_tensor = torch.tensor(
                    instruction_ids, device=device).unsqueeze(0)
                context_size = instruction_tensor.size(1)
                max_new_tokens = 30

                for _ in range(max_new_tokens):
                    with torch.no_grad():
                        logits = model(instruction_tensor)
                        logits = logits[:, -1, :]
                        logits = logits / 0.7
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        instruction_tensor = torch.cat(
                            [instruction_tensor, next_token], dim=1)
                        if next_token.item() == tokenizer.encode("<|endoftext|>")[0]:
                            break
                generated_text = tokenizer.decode(
                    instruction_tensor[0].cpu().tolist())
                print(f"\nExample {i+1}:")
                print(f"Generated: {generated_text}")
                print("-" * 50)
            else:
                print(f"\nExample {i+1}: No '### Response:' found in input")
                print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a GPT-2 model from scratch.")
    parser.add_argument("--classification_ft", action='store_true',
                        help="Run classification fine-tuning.")
    parser.add_argument("--instruction_ft", action='store_true',
                        help="Run instruction fine-tuning.")
    # Hyperparameter arguments
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay for optimizer (default: 0.1)")
    parser.add_argument("--eval_freq", type=int, default=50,
                        help="Evaluation frequency during training (default: 50)")
    parser.add_argument("--eval_iter", type=int, default=5,
                        help="Number of iterations for evaluation (default: 5)")
    # Resume training
    parser.add_argument("--resume", type=str,
                        help="Resume from checkpoint path")
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    # --- Data Loading ---
    if args.instruction_ft:
        print("Loading instruction fine-tuning data...")
        # Validate instruction data files
        if not os.path.exists("data/instruction_train.json"):
            print(
                "Error: Instruction training data file 'data/instruction_train.json' not found!")
            exit(1)
        if not os.path.exists("data/instruction_val.json"):
            print(
                "Error: Instruction validation data file 'data/instruction_val.json' not found!")
            exit(1)
        train_loader = create_instruction_dataloader(
            "data/instruction_train.json", tokenizer, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = create_instruction_dataloader(
            "data/instruction_val.json", tokenizer, batch_size=args.batch_size, shuffle=False, drop_last=False)

    elif args.classification_ft:
        print("Loading classification data...")
        # Validate classification data files
        if not os.path.exists("data/train.csv"):
            print("Error: Classification training data file 'data/train.csv' not found!")
            exit(1)
        if not os.path.exists("data/validation.csv"):
            print(
                "Error: Classification validation data file 'data/validation.csv' not found!")
            exit(1)
        train_loader = create_spam_dataloader(
            "data/train.csv", batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = create_spam_dataloader(
            "data/validation.csv", batch_size=args.batch_size, shuffle=False)
    else:
        print("Error: Please specify either --instruction_ft or --classification_ft")
        exit(1)

    # --- Model and Optimizer ---
    start_epoch = 0  # Initialize start_epoch for both paths

    if args.instruction_ft:
        print("Setting up instruction fine-tuning model...")
        model = GPTModel(GPT2_CONFIG_124M).to(device)
        model = load_weights_from_gpt2(model, hf_model_name="gpt2")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # Resume from checkpoint if specified
        if args.resume:
            if os.path.exists(args.resume):
                print(f"Resuming from checkpoint: {args.resume}")
                checkpoint = torch.load(args.resume, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(
                            checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint.get('epoch', 0)
                    print(f"Resumed from epoch {start_epoch}")
                else:
                    model.load_state_dict(checkpoint)
                    print("Loaded model weights from checkpoint")
            else:
                print(
                    f"Warning: Checkpoint file '{args.resume}' not found. Starting from scratch.")

    elif args.classification_ft:
        print("Setting up classification model...")
        model = GPTSpamClassification(GPT2_CONFIG_124M).to(device)
        model.gpt_body = load_weights_from_gpt2(
            model.gpt_body, hf_model_name="gpt2"
        )

        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreezing specific layers
        # Training the last Transformer block
        for param in model.gpt_body.trf_blocks[-1].parameters():
            param.requires_grad = True
        # Training Final Layer Norm
        for param in model.gpt_body.final_norm.parameters():
            param.requires_grad = True
        # Training the Classification Head
        for param in model.classification_head.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # Resume from checkpoint if specified
        if args.resume:
            if os.path.exists(args.resume):
                print(f"Resuming from checkpoint: {args.resume}")
                checkpoint = torch.load(args.resume, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(
                            checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint.get('epoch', 0)
                    print(f"Resumed from epoch {start_epoch}")
                else:
                    model.load_state_dict(checkpoint)
                    print("Loaded model weights from checkpoint")
            else:
                print(
                    f"Warning: Checkpoint file '{args.resume}' not found. Starting from scratch.")

    # --- Fine-Tuning ---
    print(f"Starting training on device: {device}")
    print(f"Training parameters:")
    print(f"  - Epochs: {args.num_epochs} (starting from epoch {start_epoch})")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"  - Eval frequency: {args.eval_freq}")
    print(f"  - Eval iterations: {args.eval_iter}")

    start_time = time.time()

    if args.instruction_ft:
        print("Starting instruction fine-tuning...")
        with open("data/instruction_val.json", "r") as f:
            val_json_data = json.load(f)
        start_context = format_input(val_json_data[0])
        print(f"Start context: {start_context}")
        train_losses, val_losses, track_tokens_seen = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, device=device, num_epochs=args.num_epochs,
            eval_freq=args.eval_freq, eval_iter=args.eval_iter,
            start_context=start_context, tokenizer=tokenizer,
            start_epoch=start_epoch
        )
    elif args.classification_ft:
        print("Starting classification fine-tuning...")
        train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            eval_freq=args.eval_freq,
            eval_iter=args.eval_iter,
            start_epoch=start_epoch
        )

    end_time = time.time()
    print(
        f"Finished fine-tuning in {(end_time - start_time) / 60:.2f} minutes.")

    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared.")

    # --- Save Model ---
    if args.instruction_ft:
        checkpoint_path = "checkpoints/finetuned_instruction.pth"
    elif args.classification_ft:
        checkpoint_path = "checkpoints/finetuned_spam.pth"

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save with additional metadata
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.num_epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': GPT2_CONFIG_124M,
        'args': vars(args)
    }

    if args.classification_ft:
        checkpoint_data['train_accs'] = train_accs
        checkpoint_data['val_accs'] = val_accs
        checkpoint_data['examples_seen'] = examples_seen
    else:
        checkpoint_data['track_tokens_seen'] = track_tokens_seen

    torch.save(checkpoint_data, checkpoint_path)
    print(f"Model and training state saved to {checkpoint_path}")

    # Also saving just the model weights for easier loading in inference
    model_only_path = checkpoint_path.replace('.pth', '_model_only.pth')
    torch.save(model.state_dict(), model_only_path)
    print(f"Model weights only saved to {model_only_path}")

    # --- Save Plots ---
    plot_dir = "training_results"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if args.instruction_ft:
        loss_plot_path = os.path.join(plot_dir, "instruction_loss_plot.png")
        epochs_seen = torch.linspace(
            0, args.num_epochs, len(train_losses)).tolist()
        plot_losses(epochs_seen, track_tokens_seen,
                    train_losses, val_losses, loss_plot_path)

        if os.path.exists("data/instruction_test.json"):
            print("Evaluating final instruction model on test set...")
            test_loader = create_instruction_dataloader(
                "data/instruction_test.json", tokenizer, batch_size=args.batch_size, shuffle=False, drop_last=False)
            evaluate_instruction_model(
                model, test_loader, device, tokenizer, num_examples=3)
        else:
            print("Instruction test file not found. Skipping test evaluation.")

    elif args.classification_ft:
        # Loss plot
        loss_plot_path = os.path.join(plot_dir, "spam_loss_plot.png")
        epochs_tensor = torch.linspace(0, args.num_epochs, len(train_losses))
        examples_seen_tensor = torch.linspace(
            0, examples_seen, len(train_losses))
        plot_values(epochs_tensor, examples_seen_tensor,
                    train_losses, val_losses, loss_plot_path)

        # Accuracy plot
        acc_plot_path = os.path.join(plot_dir, "spam_accuracy_plot.png")
        epochs_tensor_acc = torch.linspace(
            0, args.num_epochs, len(train_accs))
        examples_seen_tensor_acc = torch.linspace(
            0, examples_seen, len(train_accs))
        plot_values(
            epochs_tensor_acc, examples_seen_tensor_acc,
            train_accs, val_accs, acc_plot_path, label="accuracy"
        )

        # Test evaluation
        if os.path.exists("data/test.csv"):
            print("Evaluating final model on the test set...")
            test_loader = create_spam_dataloader(
                "data/test.csv", batch_size=args.batch_size, shuffle=False)
            test_accuracy = calc_accuracy_loader(test_loader, model, device)
            print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
        else:
            print("Test file not found. Skipping test evaluation.")

    print("Fine-tuning completed successfully!")
