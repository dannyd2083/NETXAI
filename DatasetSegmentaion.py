import pandas as pd
import os
from sklearn.model_selection import train_test_split


def split_malicious_and_normal(malicious_csv, normal_csv, savedir):
    """Split malicious and normal traffic separately
    training set(70%)
    validation set(10%)
    testing set(20%)
    """

    # Load CSV files
    mal_df = pd.read_csv(malicious_csv)
    normal_df = pd.read_csv(normal_csv)

    # Assign labels: 1 = Malicious traffic, 0 = Normal traffic
    mal_df["Label"] = 1
    normal_df["Label"] = 0

    # Split malicious traffic (70% train, 10% validation, 20% test)
    mal_train, mal_temp = train_test_split(mal_df, test_size=0.3, random_state=42, shuffle=True)
    mal_val, mal_test = train_test_split(mal_temp, test_size=2/3, random_state=42, shuffle=True)

    # Split normal traffic (70% train, 10% validation, 20% test)
    normal_train, normal_temp = train_test_split(normal_df, test_size=0.3, random_state=42, shuffle=True)
    normal_val, normal_test = train_test_split(normal_temp, test_size=2/3, random_state=42, shuffle=True)

    # Combine training, validation, and test sets
    train_df = pd.concat([mal_train, normal_train], ignore_index=True).sample(frac=1, random_state=42)
    val_df = pd.concat([mal_val, normal_val], ignore_index=True).sample(frac=1, random_state=42)
    test_df = pd.concat([mal_test, normal_test], ignore_index=True).sample(frac=1, random_state=42)

    # Save the split datasets
    train_path = os.path.join(savedir, "train_set.csv")
    val_path = os.path.join(savedir, "val_set.csv")
    test_path = os.path.join(savedir, "test_set.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Dataset splitting completed!")
    print(f"Training set: {len(train_df)} samples (Malicious: {len(mal_train)}, Normal: {len(normal_train)})")
    print(f"Validation set: {len(val_df)} samples (Malicious: {len(mal_val)}, Normal: {len(normal_val)})")
    print(f"Test set: {len(test_df)} samples (Malicious: {len(mal_test)}, Normal: {len(normal_test)})")


if __name__ == "__main__":
    # Execute the dataset splitting
    savedir = "Datasets"
    os.makedirs(savedir, exist_ok=True)
    split_malicious_and_normal("Feature_CSVs/malicious_feature.csv", "Feature_CSVs/normal_feature.csv", savedir)
