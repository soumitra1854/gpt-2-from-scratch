import torch
from transformers import GPT2LMHeadModel
from model import GPTModel
from utils import GPT2_CONFIG_124M

output_file = "OurModel_vs_GPT-2-124M.txt"

with open(output_file, "w") as f:
    f.write("--- Initializing models ---\n")

    # 1. Initialize your custom GPTModel
    custom_model = GPTModel(GPT2_CONFIG_124M)
    custom_sd = custom_model.state_dict()
    custom_keys = list(custom_sd.keys())

    # 2. Initialize the pre-trained Hugging Face GPT-2 model
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_sd = hf_model.state_dict()
    hf_keys = list(hf_sd.keys())

    f.write("\n--- Custom Model Keys ---\n")
    f.write("Custom Model Configuration:\n" + str(GPT2_CONFIG_124M) + "\n")
    for key in custom_keys:
        f.write(key + "\n")
    f.write(f"\nTotal keys in custom model: {len(custom_keys)}\n")

    f.write("\n--- Hugging Face Model Keys ---\n")
    f.write("Hugging Face Model Configuration:\n" + str(hf_model.config) + "\n")
    for key in hf_keys:
        f.write(key + "\n")
    f.write(f"\nTotal keys in Hugging Face model: {len(hf_keys)}\n")

    are_weights_tied = torch.equal(hf_sd['lm_head.weight'], hf_sd['transformer.wte.weight'])
    f.write(f"\nAre lm_head and wte weights tied in the HF model? {are_weights_tied}\n")
