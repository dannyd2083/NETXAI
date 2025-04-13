import os
import shutil
import pandas as pd

def main(input_dir="datasets/CTU13-CSV-Dataset", output_dir="features/cicflowmeter"):
    os.makedirs(output_dir, exist_ok=True)

    input_attack = os.path.join(input_dir, "CTU13_Attack_Traffic.csv")
    input_normal = os.path.join(input_dir, "CTU13_Normal_Traffic.csv")

    output_attack = os.path.join(output_dir, "cic_malicious.csv")
    output_normal = os.path.join(output_dir, "cic_normal.csv")

    # Copy and rename files
    if os.path.exists(input_attack):
        shutil.copy(input_attack, output_attack)
        print("Copied: {}".format(output_attack))
    else:
        print("!!!!Attack traffic file not found!!!!")

    if os.path.exists(input_normal):
        shutil.copy(input_normal, output_normal)
        print("Copied: {}".format(output_normal))
    else:
        print("!!!!Normal traffic file not found!!!!")

    # Count rows in each file
    for fpath, label in [(output_normal, "normal"), (output_attack, "malicious")]:
        if os.path.isfile(fpath):
            df = pd.read_csv(fpath)
            print("{} CIC features: {} samples".format(label.capitalize(), len(df)))

if __name__ == "__main__":
    main()
