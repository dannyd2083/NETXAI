import pandas as pd
import os
from sklearn.model_selection import train_test_split


def split_and_save(malicious_csv, normal_csv, output_dir):
    """
    Split malicious and normal traffic separately into:
    - training set (70%)
    - validation set (10%)
    - testing set (20%)
    Save combined and shuffled outputs to output_dir.
    """
    mal_df = pd.read_csv(malicious_csv)
    normal_df = pd.read_csv(normal_csv)

    mal_df["Label"] = 1
    normal_df["Label"] = 0

    # Split each class
    mal_train, mal_temp = train_test_split(mal_df, test_size=0.3, random_state=42, shuffle=True)
    mal_val, mal_test = train_test_split(mal_temp, test_size=2/3, random_state=42, shuffle=True)

    normal_train, normal_temp = train_test_split(normal_df, test_size=0.3, random_state=42, shuffle=True)
    normal_val, normal_test = train_test_split(normal_temp, test_size=2/3, random_state=42, shuffle=True)

    # Combine & shuffle
    train_df = pd.concat([mal_train, normal_train], ignore_index=True).sample(frac=1, random_state=42)
    val_df = pd.concat([mal_val, normal_val], ignore_index=True).sample(frac=1, random_state=42)
    test_df = pd.concat([mal_test, normal_test], ignore_index=True).sample(frac=1, random_state=42)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # Print stats
    print("  Datasets successfully saved to: {}".format(output_dir))
    print("  Training set: {} samples (Malicious: {}, Normal: {})".format(len(train_df), len(mal_train), len(normal_train)))
    print("  Validation set: {} samples (Malicious: {}, Normal: {})".format(len(val_df), len(mal_val), len(normal_val)))
    print("  Test set: {} samples (Malicious: {}, Normal: {})".format(len(test_df), len(mal_test), len(normal_test)))
    print("--------------------------------------------------")


def main(feature_set="basic"):
    if feature_set == "basic":
        print("-------- Splitting feature set: basic --------")
        split_and_save(
            "features/basic/basic_malicious.csv",
            "features/basic/basic_normal.csv",
            "data_splits/basic"
        )
    else:
        print("-------- Splitting feature set --------")
        split_and_save(
            "features/cicflowmeter/cic_malicious.csv",
            "features/cicflowmeter/cic_normal.csv",
            "data_splits/cicflowmeter"
        )



if __name__ == "__main__":
    main(feature_set="basic")