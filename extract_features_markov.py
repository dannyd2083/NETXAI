import pandas as pd
import numpy as np
import os


def load_data(csv_path):
    """
    Read Traffic Data from:
    Classified_CSVs/malicious_traffic.csv
    Classified_CSVs/normal_traffic.csv
    """
    df = pd.read_csv(csv_path, parse_dates=["StartTime"])
    
    # Calculate duration
    flow_group = ["SrcAddr", "DstAddr", "Sport", "Dport", "Proto"]
    df["TotalDur"] = df.groupby(flow_group)["Dur"].transform("sum")

    return df

def infer_connection_state(df):
    """
    From Label deduce connection state
    """
    def get_state(label):
        if "TCP" in label:
            if "Established" in label:
                return "Handshake Completed"
            else:
                return "Handshake Not Completed"
        elif "UDP" in label:
            return "No Handshake (UDP)"
        return "Unknown"

    df["ConnectionState"] = df["Label"].apply(get_state)
    return df

def compute_transition_matrix(df):
    """
    Calculate state transform matrix based on TotBytes
    """
    states = [200, 800, np.inf]  # define status boundary
    transition_matrix = np.zeros((3, 3))

    prev_state = None
    for bytes_count in df["TotBytes"]:
        if bytes_count < states[0]:
            state = 0
        elif bytes_count < states[1]:
            state = 1
        else:
            state = 2

        if prev_state is not None:
            transition_matrix[prev_state, state] += 1
        prev_state = state

    # Normalization
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

    return transition_matrix

def extract_features(df):
    """
    Extract Statistics features
    """
    # Compute packet rate
    df["PktRate"] = df["TotPkts"] / df["Dur"].replace(0, np.nan)
    # Compute byte rate
    df["ByteRate"] = df["TotBytes"] / df["Dur"].replace(0, np.nan)
    # Compute Flow Portion
    df["SrcRatio"] = df["SrcBytes"] / df["TotBytes"].replace(0, np.nan)

    # Compute status transform matrix
    transition_matrix = compute_transition_matrix(df)
    for i in range(3):
        for j in range(3):
            df[f"tm_{i}_{j}"] = transition_matrix[i, j]

    return df

def filter_flows(df):
    """
    Filter flows based on the rule defined in the paper: only keep duration < 30 min
    """
    df = df[df["TotalDur"] < 30 * 60]
    # df = infer_connection_state(df)

    # # Only keep connections that complete TCP 3-way handshake
    # df = df[(df["ConnectionState"] == "Handshake Completed") | (df["ConnectionState"] == "No Handshake (UDP)")]

    return df

def process_csv(csv_path, output_csv):
    """
    Run Feature Execution
    """
    output_dir = "Feature_CSVs"
    output_path = os.path.join(output_dir, output_csv)
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(csv_path)
    df = filter_flows(df)
    df = extract_features(df)

    # Output columns
    feature_cols = ["PktRate", "ByteRate", "SrcRatio"] + [f"tm_{i}_{j}" for i in range(3) for j in range(3)]
    df[feature_cols].to_csv(output_path, index=False)
    
    print(f"Processed data saved to {output_csv}")


if __name__ == "__main__":
    process_csv("Classified_CSVs/normal_traffic.csv", "features_normal_markov.csv")
    process_csv("Classified_CSVs/malicious_traffic.csv", "features_malicious_markov.csv")
