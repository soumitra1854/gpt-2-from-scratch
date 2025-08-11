import torch
import os
import time
import argparse
import tiktoken
from utils import (
    Custom_GPT2_config,
    GPT2_CONFIG_124M,
    generate,
    calc_loss_loader,
    calc_loss_batch,
    text_to_token_ids,
    token_ids_to_text,
    load_weights_from_gpt2
)
from model import GPTModel, GPTSpamClassification
from dataloader import create_dataloader, create_spam_dataloader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")


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
    return train_loss / eval_iter, val_loss / eval_iter


def plot_values(epochs_seen, examples_seen, train_values, val_values, save_path, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.",
             label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    # Invisible plot for aligning ticks
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(save_path)
    plt.close()
    print(f"{label.capitalize()} plot saved to {save_path}")


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
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen


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


def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
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


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a GPT model.")
    parser.add_argument("--file_path", type=str, default="data/the-verdict.txt",
                        help="Path to the training text file.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float,
                        default=5e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/trained_model.pth",
                        help="Path to save the model checkpoint.")
    parser.add_argument("--classification", action='store_true',
                        help="If set, train a classification model instead of a language model.")
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)

    # --- Data Loading ---
    if args.classification:
        train_loader = create_spam_dataloader(
            "data/train.csv", batch_size=8, shuffle=True, drop_last=True)
        val_loader = create_spam_dataloader(
            "data/validation.csv", batch_size=8, shuffle=False)
    else:
        with open(args.file_path, "r", encoding='utf-8') as f:
            text_data = f.read()

        train_ratio = 0.90
        split_idx = int(train_ratio * len(text_data))
        train_text = text_data[:split_idx]
        val_text = text_data[split_idx:]

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

    # --- Model and Optimizer ---
    if args.classification:
        model = GPTSpamClassification(GPT2_CONFIG_124M).to(device)
        model.gpt_body = load_weights_from_gpt2(
            model.gpt_body, hf_model_name="gpt2"
        )
        for param in model.parameters():
            param.requires_grad = False

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
            lr=0.0001,
            weight_decay=0.1
        )
    else:
        # Initialize the GPT model for text generation
        model = GPTModel(Custom_GPT2_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    tokenizer = tiktoken.get_encoding("gpt2")

    # --- Training ---
    start_time = time.time()
    if args.classification:
        train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=5,
            eval_freq=50,
            eval_iter=5
        )
    else:
        train_losses, val_losses, track_tokens_seen = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            eval_freq=5,
            eval_iter=5,
            start_context="Every effort moves you",
            tokenizer=tokenizer
        )
    end_time = time.time()
    print(f"Training finished in {(end_time - start_time) / 60:.2f} minutes.")

    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(model.state_dict(), args.checkpoint_path)
    print(f"Model saved to {args.checkpoint_path}")

    plot_dir = "training_results"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if args.classification:
        loss_plot_path = os.path.join(plot_dir, "loss_plot.png")
        epochs_tensor = torch.linspace(0, 5, len(train_losses))
        examples_seen_tensor = torch.linspace(
            0, examples_seen, len(train_losses))
        plot_values(epochs_tensor, examples_seen_tensor,
                    train_losses, val_losses, loss_plot_path)
        acc_plot_path = os.path.join(plot_dir, "accuracy_plot.png")
        epochs_tensor_acc = torch.linspace(0, args.num_epochs, len(train_accs))
        examples_seen_tensor_acc = torch.linspace(
            0, examples_seen, len(train_accs))
        plot_values(
            epochs_tensor_acc, examples_seen_tensor_acc,
            train_accs, val_accs, acc_plot_path, label="accuracy"
        )
        print("Evaluating final model on the test set...")
        test_loader = create_spam_dataloader(
            "data/test.csv",
            batch_size=8,
            shuffle=False
        )
        test_accuracy = calc_accuracy_loader(test_loader, model, device)
        print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
    else:
        plot_path = os.path.join(plot_dir, "loss_plot.png")
        epochs_seen = torch.linspace(
            0, args.num_epochs, len(train_losses)).tolist()
        plot_losses(epochs_seen, track_tokens_seen,
                    train_losses, val_losses, plot_path)
