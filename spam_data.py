import pandas as pd
import urllib.request
import zipfile
import os
from pathlib import Path


def download_and_unzip(url, zip_path, extracted_path, data_filename):
    """Downloads and unzips the SMS Spam Collection dataset."""
    data_file_path = Path(extracted_path) / data_filename
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download.")
        return
    os.makedirs(extracted_path, exist_ok=True)
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")
    os.remove(zip_path)


def create_balanced_dataset(df):
    """Balances the dataset by undersampling the 'ham' class."""
    num_spam = df[df["label"] == "spam"].shape[0]
    ham_subset = df[df["label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["label"] == "spam"]])


def random_split(df, train_frac, validation_frac):
    """Shuffles and splits the DataFrame into train, validation, and test sets."""
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df


if __name__ == "__main__":
    URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    ZIP_PATH = "sms_spam_collection.zip"
    EXTRACTED_PATH = "data"
    DATA_FILENAME = "SMSSpamCollection.tsv"

    download_and_unzip(URL, ZIP_PATH, EXTRACTED_PATH, DATA_FILENAME)

    # Load the data into a pandas DataFrame
    df = pd.read_csv(
        Path(EXTRACTED_PATH) / DATA_FILENAME,
        sep="\t",
        header=None,
        names=["label", "text"]
    )

    balanced_df = create_balanced_dataset(df)
    print(f"Label Counts:\n{balanced_df['label'].value_counts()}")
    balanced_df["label"] = balanced_df["label"].map({"ham": 0, "spam": 1})

    # Split the data: 0.7 train, 0.1 validation, 0.2 test
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    # Save the final CSV files
    train_df.to_csv(Path(EXTRACTED_PATH) / "train.csv", index=None)
    validation_df.to_csv(Path(EXTRACTED_PATH) / "validation.csv", index=None)
    test_df.to_csv(Path(EXTRACTED_PATH) / "test.csv", index=None)

    print("Data preparation complete.")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(validation_df)}")
    print(f"Test set size: {len(test_df)}")
