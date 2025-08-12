import json
import os
import urllib.request
from pathlib import Path

def download_and_load_json(file_path, url):
    """Downloads a JSON file from a URL and loads it."""
    if not os.path.exists(file_path):
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        print(f"{file_path} already exists. Skipping download.")
    
    with open(file_path, "r") as file:
        data = json.load(file)
    
    return data

if __name__ == "__main__":
    # Define file paths and URL
    file_path = "data/instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    os.makedirs("data", exist_ok=True)
    data = download_and_load_json(file_path, url)
    print(f"Number of entries: {len(data)}") # Expected: 1100

    # Splitting the dataset into train: 85%, test: 10%, val: 5%
    train_portion = int(len(data) * 0.85) 
    test_portion = int(len(data) * 0.1)  
    val_portion = len(data) - train_portion - test_portion
    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    # Saving the split files
    train_path = Path("data") / "instruction_train.json"
    val_path = Path("data") / "instruction_val.json"
    test_path = Path("data") / "instruction_test.json"
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=4)
    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=4)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=4)

    print("Data preparation complete.")
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")