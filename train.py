import torch
import argparse
import os
import time
import tiktoken
from utils import (
    Custom_GPT2_config,
    generate,
    calc_loss_loader,
    calc_loss_batch,
    text_to_token_ids,
    token_ids_to_text
)
from model import GPTModel
from dataloader import create_dataloader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(epochs_seen, train_losses, label="Training loss", linewidth=2)
    ax1.plot(epochs_seen, val_losses, linestyle="-.",
             label="Validation loss", linewidth=2)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen", fontsize=12)
    plt.title("Training and Validation Loss over Epochs",
              fontsize=16, fontweight='bold', color='blue', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {save_path}")


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, input_ids=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"Sample generation: {decoded_text.replace(chr(10), ' ')}")
    model.train()


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer, start_epoch=0):
    train_losses, val_losses, track_tokens_seen = [], [], []
    num_tokens_seen, global_step = 0, -1

    print(f"Starting training from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_start_time = time.time()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            # Gradient clipping for stability
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            num_tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(num_tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

        # Printing a sample response after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 model from scratch.")

    # Data arguments
    parser.add_argument("--file_path", type=str, default="data/the-verdict.txt",
                        help="Path to the training text file.")
    parser.add_argument("--train_ratio", type=float, default=0.90,
                        help="Ratio of data to use for training (default: 0.90)")

    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training.")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay for optimizer (default: 0.1)")
    parser.add_argument("--eval_freq", type=int, default=5,
                        help="Evaluation frequency during training (default: 5)")
    parser.add_argument("--eval_iter", type=int, default=5,
                        help="Number of iterations for evaluation (default: 5)")

    # Model and checkpoint arguments
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/trained_model.pth",
                        help="Path to save the model checkpoint.")
    parser.add_argument("--resume", type=str,
                        help="Resume training from checkpoint")
    parser.add_argument("--start_context", type=str, default="Every effort moves you",
                        help="Starting context for sample generation")

    # Output arguments
    parser.add_argument("--plot_dir", type=str, default="training_results",
                        help="Directory to save training plots")
    parser.add_argument("--save_freq", type=int, default=0,
                        help="Save checkpoint every N epochs (0 = only at end)")

    args = parser.parse_args()

    # --- Validation ---
    if not os.path.exists(args.file_path):
        print(f"Error: Training file '{args.file_path}' not found!")
        print("Please ensure the training data file exists.")
        exit(1)

    if args.train_ratio <= 0 or args.train_ratio >= 1:
        print("Error: train_ratio must be between 0 and 1")
        exit(1)

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    print(f"Using device: {device}")
    print(f"Training parameters:")
    print(f"  - File: {args.file_path}")
    print(f"  - Epochs: {args.num_epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(
        f"  - Train/Val split: {args.train_ratio:.1%}/{1-args.train_ratio:.1%}")

    # --- Data Loading ---
    print("Loading and preprocessing data...")
    try:
        with open(args.file_path, "r", encoding='utf-8') as f:
            text_data = f.read()
    except UnicodeDecodeError:
        print("Warning: UTF-8 decoding failed, trying with latin-1 encoding")
        with open(args.file_path, "r", encoding='latin-1') as f:
            text_data = f.read()

    if len(text_data) == 0:
        print("Error: Training file is empty!")
        exit(1)

    print(f"Loaded text with {len(text_data):,} characters")

    split_idx = int(args.train_ratio * len(text_data))
    train_text = text_data[:split_idx]
    val_text = text_data[split_idx:]

    print(f"Training text: {len(train_text):,} characters")
    print(f"Validation text: {len(val_text):,} characters")

    train_loader = create_dataloader(
        train_text,
        batch_size=args.batch_size,
        context_size=Custom_GPT2_config["context_length"],
        stride=Custom_GPT2_config["context_length"]
    )
    val_loader = create_dataloader(
        val_text,
        batch_size=args.batch_size,
        context_size=Custom_GPT2_config["context_length"],
        stride=Custom_GPT2_config["context_length"]
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # --- Model and Optimizer ---
    print("Initializing model and optimizer...")
    model = GPTModel(Custom_GPT2_config).to(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(
        f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # --- Resume Training if Specified ---
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("Loaded model state from checkpoint")

                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(
                            checkpoint['optimizer_state_dict'])
                        print("Loaded optimizer state from checkpoint")

                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch']
                        print(f"Resuming from epoch {start_epoch}")

                    if 'train_losses' in checkpoint:
                        print("Previous training history available in checkpoint")
                else:
                    model.load_state_dict(checkpoint)
                    print("Loaded model weights from checkpoint")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model weights from checkpoint")
        else:
            print(
                f"Warning: Checkpoint file '{args.resume}' not found. Starting from scratch.")

    # --- Training ---
    print("\nStarting training...")
    start_time = time.time()

    try:
        train_losses, val_losses, track_tokens_seen = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            eval_freq=args.eval_freq,
            eval_iter=args.eval_iter,
            start_context=args.start_context,
            tokenizer=tokenizer,
            start_epoch=start_epoch
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        # Save interrupted training state
        interrupted_path = args.checkpoint_path.replace(
            '.pth', '_interrupted.pth')
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': start_epoch,  # Current epoch when interrupted
            'config': Custom_GPT2_config,
            'args': vars(args)
        }
        torch.save(checkpoint_data, interrupted_path)
        print(f"Interrupted state saved to {interrupted_path}")
        exit(0)

    end_time = time.time()
    training_time = (end_time - start_time) / 60
    print(f"\nTraining completed in {training_time:.2f} minutes")
    print(
        f"Average time per epoch: {training_time/args.num_epochs:.2f} minutes")

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")

    # --- Save Model ---
    print("Saving final model...")
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save comprehensive checkpoint
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.num_epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'track_tokens_seen': track_tokens_seen,
        'config': Custom_GPT2_config,
        'args': vars(args),
        'training_time_minutes': training_time
    }
    torch.save(checkpoint_data, args.checkpoint_path)
    print(f"Full checkpoint saved to {args.checkpoint_path}")

    # Also save model-only version for inference
    model_only_path = args.checkpoint_path.replace('.pth', '_model_only.pth')
    torch.save(model.state_dict(), model_only_path)
    print(f"Model weights saved to {model_only_path}")

    # --- Generate Training Plots ---
    print("Generating training plots...")
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    plot_path = os.path.join(args.plot_dir, "train_loss_plot.png")
    epochs_seen = torch.linspace(
        0, args.num_epochs, len(train_losses)).tolist()
    plot_losses(epochs_seen, track_tokens_seen,
                train_losses, val_losses, plot_path)

    # --- Final Summary ---
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Training file: {args.file_path}")
    print(f"Epochs completed: {args.num_epochs}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Total tokens processed: {track_tokens_seen[-1]:,}")
    print(f"Training time: {training_time:.2f} minutes")
    print(f"Model saved to: {args.checkpoint_path}")
    print(f"Plots saved to: {args.plot_dir}")

    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nTraining completed successfully! 🎉")
