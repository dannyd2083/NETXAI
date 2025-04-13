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
    parser.add_argument("--feature_set", choices=["basic", "cicflowmeter"], required=True)
    parser.add_argument("--model", nargs="*", choices=["xgb", "rf", "lgbm", "knn", "lr"])
    args = parser.parse_args()

    feature_set = args.feature_set
    selected_models = args.model

    print(f"\n Running pipeline for: {feature_set} feature set")

    print("\n Step 1: Downloading datasets... It may take long time because the dataset is huge!")
    download_and_setup_data.main()

    print("\n Step 2: Processing raw .binetflow files...")
    process_raw_binetflow.main()

    print("\n Step 3: Extracting basic features...")
    extract_features_basic.main()

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

    for key, module in model_map.items():
        if selected_models is None or key in selected_models:
            print(f"\n Training model: {key.upper()}")
            module.main(feature_set)

    print("\n📊 Step 10: Generating radar chart...")
    radar_plot_from_results.main()

    print(f"\n Pipeline completed for: {feature_set}\n")


if __name__ == "__main__":
    main()