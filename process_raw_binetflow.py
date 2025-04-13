import pandas as pd
import glob
import os


# Convert .binetflow files into intermediate CSVs
def process_binetflow_to_csv(input_pattern="datasets/ctu13/*/*/capture*.binetflow", output_dir="datasets/ctu13/processed_binetflow"):
    file_list = sorted(glob.glob(input_pattern))
    if os.path.exists(output_dir) or not file_list:
        print("Skipping conversion. Output directory '{}' already exists or no .binetflow files found.".format(output_dir))
        return

    os.makedirs(output_dir, exist_ok=True)

    for file in file_list:
        dataset_number = os.path.normpath(file).split(os.sep)[-2]
        df = pd.read_csv(file)
        output_file = os.path.join(output_dir, f"capture_{dataset_number}.csv")
        df.to_csv(output_file, index=False)
        print("Processed and saved: {}".format(output_file))

# Classify flows into malicious and normal, and save separate CSVs
def classify_traffic(input_pattern="datasets/ctu13/processed_binetflow/capture_*.csv", output_dir="datasets/ctu13/classified_binetflow"):
    file_list = sorted(glob.glob(input_pattern))
    if os.path.exists(output_dir) or not file_list:
        print("Skipping classification. Output directory '{}' already exists or no CSV files found.".format(output_dir))
        return

    os.makedirs(output_dir, exist_ok=True)

    malicious_traffic = []
    normal_traffic = []

    for file in file_list:
        df = pd.read_csv(file)
        if 'Label' not in df.columns:
            print("Warning: 'Label' column not found in {}".format(file))
            continue

        malicious_df = df[df['Label'].str.contains("botnet", case=False, na=False)]
        normal_df = df[df['Label'].str.contains("Normal", case=False, na=False)]

        malicious_traffic.append(malicious_df)
        normal_traffic.append(normal_df)

    malicious_out = os.path.join(output_dir, "malicious.csv")
    normal_out = os.path.join(output_dir, "normal.csv")

    pd.concat(malicious_traffic, ignore_index=True).to_csv(malicious_out, index=False)
    pd.concat(normal_traffic, ignore_index=True).to_csv(normal_out, index=False)

    print("Saved classified traffic:")
    print("  --> {}".format(malicious_out))
    print("  --> {}".format(normal_out))

# Print number of rows in each classified file
def count_traffic_rows(classified_dir="datasets/ctu13/classified_binetflow"):
    for label in ["malicious", "normal"]:
        path = os.path.join(classified_dir, f"{label}.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            print("{} {} traffic rows in total.".format(len(df), label))

if __name__ == "__main__":
    process_binetflow_to_csv()
    classify_traffic()
    count_traffic_rows()
