import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd


class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, context_size, stride):
        self.input_ids = []
        self.target_ids = []
        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - context_size, stride):
            input_chunk = token_ids[i:i + context_size]
            target_chunk = token_ids[i + 1: i + context_size + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size=4, context_size=256,
                      stride=128, shuffle=True, drop_last=True,
                      num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, context_size, stride)
    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.encoded_texts = [
            self.tokenizer.encode(text) for text in self.data['text']
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # trimming the exctra words
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]

        # padding
        self.pad_token_id = pad_token_id
        self.encoded_texts = [
            encoded_text + [self.pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
        label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)
        return (input_ids, label)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def create_spam_dataloader(csv_file, batch_size=8, max_length=None, shuffle=True, drop_last=False, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = SpamDataset(csv_file, tokenizer, max_length=max_length)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )

# Helper function to format prompts using the Alpaca style


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        with open(data_path, "r") as file:
            self.data = json.load(file)

        self.tokenizer = tokenizer
        self.encoded_texts = []

        for entry in self.data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            # adding an end-of-text token
            full_text += "<|endoftext|>"
            self.encoded_texts.append(self.tokenizer.encode(
                full_text, allowed_special={"<|endoftext|>"}))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,  device="cpu", allowed_max_length=None):
    batch_max_length = max(len(item) for item in batch) + 1
    inputs_lst, targets_lst = [], []

    for item in batch:
        padded = item + [pad_token_id] * (batch_max_length - len(item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # Mask loss for padding tokens by replacing them with ignore_index except the first
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None and len(inputs) > allowed_max_length:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def create_instruction_dataloader(json_file, tokenizer=tiktoken.get_encoding("gpt2"), batch_size=8, shuffle=False, drop_last=False, num_workers=0):
    dataset = InstructionDataset(json_file, tokenizer)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
