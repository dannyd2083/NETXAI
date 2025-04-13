import pandas as pd
import glob
import os


# Transform the 
def process_original_to_csv(file_pattern):

    # get all matched .binetflow files
    file_list = sorted(glob.glob(file_pattern))

    # dir to store csv files
    output_dir = "Processed_CSVs"
    if os.path.exists(output_dir) or file_list == []:
        return
    os.makedirs(output_dir, exist_ok=True)

    # process files
    for file in file_list:
        normalized_file = os.path.normpath(file)
        split_path = normalized_file.split(os.sep)
        dataset_number = split_path[-2] if len(split_path) > 2 else "unknown"
        df = pd.read_csv(file)
        # set output filename
        output_file = os.path.join(output_dir, f"capture_{dataset_number}.csv")
        
        # save as csv format
        df.to_csv(output_file, index=False)

        print("Processed and saved: {}".format(output_file))

def traffic_classifier(csv_path):
    file_list = sorted(glob.glob(csv_path))
    output_dir = "Classified_CSVs"
    if os.path.exists(output_dir) or file_list == []:
        return
    os.makedirs(output_dir, exist_ok=True)

    malicious_traffic = []
    normal_traffic = []

    # read csv traffic files
    for file in file_list:
        df = pd.read_csv(file)

        # ensure Label column exists
        if 'Label' not in df.columns:
            print("Warning: 'Label' column not found in {}".format(file))
            continue

        # classify traffic
        malicious_df = df[df['Label'].str.contains("botnet", case=False, na=False)]
        normal_df = df[df['Label'].str.contains("Normal", case=False, na=False)]

        malicious_traffic.append(malicious_df)
        normal_traffic.append(normal_df)

    # combine all classified data
    malicious_df_combined = pd.concat(malicious_traffic, ignore_index=True)
    normal_df_combined = pd.concat(normal_traffic, ignore_index=True)

    # save to csv file
    malicious_file = os.path.join(output_dir, "malicious_traffic.csv")
    normal_file = os.path.join(output_dir, "normal_traffic.csv")

    malicious_df_combined.to_csv(malicious_file, index=False)
    normal_df_combined.to_csv(normal_file, index=False)

    print("Saved: {}".format(malicious_file))
    print("Saved: {}".format(normal_file))


def count_traffic_row():

    malicious_file = "Classified_CSVs/malicious_traffic.csv"
    if os.path.isfile(malicious_file):
        df_1 = pd.read_csv(malicious_file)
        num_rows = len(df_1)
        print("{} malicious traffic in total.".format(num_rows))

    normal_file = "Classified_CSVs/normal_traffic.csv"
    if os.path.isfile(normal_file):
        df_2 = pd.read_csv(normal_file)
        num_rows = len(df_2)
        print("{} normal traffic in total.".format(num_rows))




if __name__ == "__main__":
    # set file pattern
    input_pattern = "CTU-13-Dataset/*/capture*.binetflow"
    process_original_to_csv(input_pattern)
    input_pattern = "Processed_CSVs/capture_*.csv"
    traffic_classifier(input_pattern)
    count_traffic_row()
