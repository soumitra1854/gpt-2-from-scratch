import torch
from transformers import GPT2LMHeadModel

GPT2_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 1024,  # context length
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": True       # Query-key-value bias
}

Custom_GPT2_config = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256,  # context length (original: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False       # Query-key-value bias (original: True)
}


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat_ids = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat_ids)


def generate(model, input_ids, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        input_ids_cond = input_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(input_ids_cond)
        # select the last token's logits for next token prediction
        logits = logits[:, -1, :]

        # New: Apply top-k filtering if specified
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(1)
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )  # (batch_size, 1)
        else:
            idx_next = torch.argmax(
                logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:
            # If end-of-sequence token is generated, stop sampling
            break

        # (batch_size, num_tokens+1)
        input_ids = torch.cat((input_ids, idx_next), dim=1)
    return input_ids


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def load_weights_from_gpt2(custom_model, hf_model_name="gpt2"):
    print(f"Loading weights from Hugging Face model: '{hf_model_name}'...")
    hf_model = GPT2LMHeadModel.from_pretrained(hf_model_name)
    hf_sd = hf_model.state_dict()  # hf_sd -> Hugging Face state_dict
    custom_sd = custom_model.state_dict()  # custom_sd -> our model's state_dict

    # Token and Positional Embeddings
    custom_sd["tok_emb.weight"].copy_(hf_sd["transformer.wte.weight"])
    custom_sd["pos_emb.weight"].copy_(hf_sd["transformer.wpe.weight"])

    # Transformer Blocks (Iterate through each layer)
    for i in range(custom_model.trf_blocks.__len__()):
        # -- LayerNorm 1--
        custom_sd[f"trf_blocks.{i}.norm1.scale"].copy_(
            hf_sd[f"transformer.h.{i}.ln_1.weight"])
        custom_sd[f"trf_blocks.{i}.norm1.shift"].copy_(
            hf_sd[f"transformer.h.{i}.ln_1.bias"])
        # -- Attention Block --
        # QKV Weights (Split and Transpose)
        q_w, k_w, v_w = hf_sd[f"transformer.h.{i}.attn.c_attn.weight"].chunk(
            3, dim=1)
        custom_sd[f"trf_blocks.{i}.att.W_query.weight"].copy_(q_w.T)
        custom_sd[f"trf_blocks.{i}.att.W_key.weight"].copy_(k_w.T)
        custom_sd[f"trf_blocks.{i}.att.W_value.weight"].copy_(v_w.T)
        # QKV Biases (Split, No Transpose)
        q_b, k_b, v_b = hf_sd[f"transformer.h.{i}.attn.c_attn.bias"].chunk(3)
        custom_sd[f"trf_blocks.{i}.att.W_query.bias"].copy_(q_b)
        custom_sd[f"trf_blocks.{i}.att.W_key.bias"].copy_(k_b)
        custom_sd[f"trf_blocks.{i}.att.W_value.bias"].copy_(v_b)
        # Output Projection (Transpose Weight, No Transpose Bias)
        custom_sd[f"trf_blocks.{i}.att.out_proj.weight"].copy_(
            hf_sd[f"transformer.h.{i}.attn.c_proj.weight"].T)
        custom_sd[f"trf_blocks.{i}.att.out_proj.bias"].copy_(
            hf_sd[f"transformer.h.{i}.attn.c_proj.bias"])
        # -- LayerNorm 2 --
        custom_sd[f"trf_blocks.{i}.norm2.scale"].copy_(
            hf_sd[f"transformer.h.{i}.ln_2.weight"])
        custom_sd[f"trf_blocks.{i}.norm2.shift"].copy_(
            hf_sd[f"transformer.h.{i}.ln_2.bias"])
        # -- Feed-Forward Block --
        # Fully Connect Layer (Transpose Weight only)
        custom_sd[f"trf_blocks.{i}.ff.layers.0.weight"].copy_(
            hf_sd[f"transformer.h.{i}.mlp.c_fc.weight"].T)
        custom_sd[f"trf_blocks.{i}.ff.layers.0.bias"].copy_(
            hf_sd[f"transformer.h.{i}.mlp.c_fc.bias"])
        # Projection Layer (Transpose Weight only)
        custom_sd[f"trf_blocks.{i}.ff.layers.2.weight"].copy_(
            hf_sd[f"transformer.h.{i}.mlp.c_proj.weight"].T)
        custom_sd[f"trf_blocks.{i}.ff.layers.2.bias"].copy_(
            hf_sd[f"transformer.h.{i}.mlp.c_proj.bias"])

    # Final LayerNorm (No Transpose)
    custom_sd["final_norm.scale"].copy_(hf_sd["transformer.ln_f.weight"])
    custom_sd["final_norm.shift"].copy_(hf_sd["transformer.ln_f.bias"])

    # Weight Tying for the Output Head
    custom_sd["out_head.weight"] = custom_sd["tok_emb.weight"]

    # Loading the modified state dictionary into our model
    custom_model.load_state_dict(custom_sd)
    print("Weights loaded successfully.")
    return custom_model
