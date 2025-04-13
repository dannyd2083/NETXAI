import pandas as pd
import numpy as np
import argparse
import os

def extract_basic_features(input_csv, output_csv, add_features):
    # Read input CSV
    df = pd.read_csv(input_csv)

    # Required columns for feature computation
    selected_columns = [
        "Dur",         # Flow duration
        "TotPkts",     # Total packets
        "TotBytes",    # Total bytes
        "SrcBytes",    # Bytes sent from source
        "Proto",       # Protocol (tcp/udp)
        "Dir"          # Direction
    ]
    
    for extra in add_features:
        selected_columns.append(extra)

    df = df[selected_columns].copy()

    # Derived features
    df["SrcRatio"] = df["SrcBytes"] / df["TotBytes"].replace(0, np.nan)
    df["PktRate"] = df["TotPkts"] / df["Dur"].replace(0, np.nan)
    df["ByteRate"] = df["TotBytes"] / df["Dur"].replace(0, np.nan)
    df.fillna(0.0, inplace=True)

    # Encode protocol: tcp -> 0, udp -> 1, other -> -1
    df["Proto"] = df["Proto"].map({"tcp": 0, "udp": 1}).fillna(-1)

    # Encode direction: "->" -> 1, "<-" -> 0, else -1
    df["Dir"] = df["Dir"].str.strip()
    df["Dir"] = df["Dir"].map({"->": 1, "<-": 0}).fillna(-1)

    # Final feature list
    feature_columns = [
        "Dur", "TotPkts", "TotBytes", "SrcBytes",
        "SrcRatio", "PktRate", "ByteRate",
        "Proto", "Dir"
    ]

    # Optional: one-hot encode Sport if included
    if "Sport" in selected_columns:
        top_ports = df["Sport"].value_counts().head(20).index
        df["Sport_cat"] = df["Sport"].apply(lambda x: str(x) if x in top_ports else "other")
        sport_dummies = pd.get_dummies(df["Sport_cat"], prefix="Sport").astype(int)
        df = pd.concat([df, sport_dummies], axis=1)
        feature_columns += list(sport_dummies.columns)

    # Save to output file
    df[feature_columns].to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")


def main(sport = False):
    extra_features = []
    if sport:
        extra_features.append("Sport")

    input_dir = "datasets/ctu13/classified_binetflow"
    output_dir = "features/basic"
    os.makedirs(output_dir, exist_ok=True)

    # Extract for normal traffic
    input_normal = os.path.join(input_dir, "normal.csv")
    output_normal = os.path.join(output_dir, "basic_normal.csv")
    extract_basic_features(input_normal, output_normal, extra_features)

    # Extract for malicious traffic
    input_malicious = os.path.join(input_dir, "malicious.csv")
    output_malicious = os.path.join(output_dir, "basic_malicious.csv")
    extract_basic_features(input_malicious, output_malicious, extra_features)

    # Count rows after feature extraction
    for fpath, label in [(output_normal, "normal"), (output_malicious, "malicious")]:
        if os.path.isfile(fpath):
            df = pd.read_csv(fpath)
            print(f"{label.capitalize()} features: {len(df)} samples")


if __name__ == "__main__":
    main()



