import pandas as pd
import numpy as np
import argparse
import os


def feature_extraction(input_csv, output_csv, addFea):
    # read original data source
    df = pd.read_csv(input_csv)

    # Extract original feature fields
    selected_columns = [
        "Dur",         # Flow duration
        "TotPkts",     # Total packets
        "TotBytes",    # Total bytes
        "SrcBytes",    # Source bytes
        "Proto",       # Protocol (TCP/UDP)
        "Dir"          # Direction (-> or <-)
    ]
    # print(addFea)
    for addfeature in addFea:
        selected_columns.append(addfeature)
    # print(selected_columns)
    df = df[selected_columns].copy()
    # Calculate SrcRatio
    df["SrcRatio"] = df["SrcBytes"] / df["TotBytes"].replace(0, np.nan)
    df["SrcRatio"] = df["SrcRatio"].fillna(0.0)

    # Calculate Packet Rate and Byte Rate
    df["PktRate"] = df["TotPkts"] / df["Dur"].replace(0, np.nan)
    df["ByteRate"] = df["TotBytes"] / df["Dur"].replace(0, np.nan)
    df["PktRate"] = df["PktRate"].fillna(0.0)
    df["ByteRate"] = df["ByteRate"].fillna(0.0)

    # Statistical Proto fields
    df["Proto"] = df["Proto"].map({"tcp": 0, "udp": 1}).fillna(-1)

    # Statistical Direction field
    df["Dir"] = df["Dir"].str.strip()
    df["Dir"] = df["Dir"].map({"->": 1, "<-": 0}).fillna(-1)


    # # Final feature columns
    feature_columns = [
        "Dur", "TotPkts", "TotBytes", "SrcBytes",
        "SrcRatio", "PktRate", "ByteRate",
        "Proto", "Dir"
    ]

    if "Sport" in selected_columns:
        # Handle Sport (source port) using top-N one-hot encoding
        top_ports = df["Sport"].value_counts().head(20).index
        df["Sport_cat"] = df["Sport"].apply(lambda x: str(x) if x in top_ports else "other")
        sport_dummies = pd.get_dummies(df["Sport_cat"], prefix="Sport").astype(int)
        df = pd.concat([df, sport_dummies], axis=1)
        feature_columns +=  list(sport_dummies.columns)
        



    # Save to output CSV
    df[feature_columns].to_csv(output_csv, index=False)
    print(f"Feature extraction completed. Output saved to {output_csv}")


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model on flow-level features.")
    parser.add_argument("--sport", action="store_true", help="Include Sport feature (default: False)")
    args = parser.parse_args()
    Sport_flag = args.sport
    # print(Sport_flag)
    
    addFea = []
    if Sport_flag:
        addFea.append("Sport")

    outdir = "Feature_CSVs"
    os.makedirs(outdir, exist_ok=True)
    input_csv = "Classified_CSVs/normal_traffic.csv"
    output_csv = outdir + "/normal_feature.csv"
    feature_extraction(input_csv, output_csv, addFea)

    input_csv = "Classified_CSVs/malicious_traffic.csv"
    output_csv = outdir + "/malicious_feature.csv"
    feature_extraction(input_csv, output_csv, addFea)

