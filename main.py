import argparse

import download_and_setup_data
import process_raw_binetflow
import extract_features_basic
import extract_features_cic
import split_dataset
import train_xgb_models as xgb
import train_rf_models as rf
import train_lgbm_models as lgbm
import train_knn_models as knn
import train_lr_models as lr
import radar_plot_from_results


def main():
    parser = argparse.ArgumentParser(description="Main pipeline runner for malicious traffic detection")
    parser.add_argument("--feature_set", choices=["basic", "cicflowmeter"], default="basic",
                        help="Choose feature set to use (default: basic)")
    parser.add_argument("--model", nargs="*", choices=["xgb", "rf", "lgbm", "knn", "lr"],
                        help="Specify which model(s) to run. Default: all. ")

    args = parser.parse_args()
    feature_set = args.feature_set
    selected_models = args.model

    print("\n Running pipeline for: {} feature set".format(feature_set))

    print("\n Step 1: Downloading datasets... (It may take a while)")
    download_and_setup_data.main()

    print("\n Step 2: Processing raw .binetflow files...")
    process_raw_binetflow.main()

    if feature_set == "basic":
        print("\n Step 3: Extracting basic features...")
        extract_features_basic.main()
    else:
        print("\n Step 3: Extracting CICFlowMeter features...")
        extract_features_cic.main()

    print("\n Step 4: Splitting dataset...")
    split_dataset.main()

    model_map = {
        "xgb": xgb,
        "rf": rf,
        "lgbm": lgbm,
        "knn": knn,
        "lr": lr
    }

    print("\n Step 5: Training models...")
    for key, module in model_map.items():
        if selected_models is None or key in selected_models:
            print(f"\n----->>>>>> Training model: {key.upper()}")
            module.main(feature_set)

    print("\n Step 6: Generating radar chart...")
    radar_plot_from_results.main(feature_set)

    print(f"\n Pipeline completed successfully for: {feature_set} feature set\n")


if __name__ == "__main__":
    main()