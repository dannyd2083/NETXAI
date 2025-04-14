import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from result_utils import save_results_to_csv
from shap_utils import generate_and_save_shap_plots
import time
import argparse


def load_split_data(split_dir):
    train_path = os.path.join(split_dir, "train.csv")
    val_path = os.path.join(split_dir, "val.csv")
    test_path = os.path.join(split_dir, "test.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=["Label"])
    y_train = df_train["Label"]
    X_val = df_val.drop(columns=["Label"])
    y_val = df_val["Label"]
    X_test = df_test.drop(columns=["Label"])
    y_test = df_test["Label"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_xgb_classifier(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    return [acc, prec, rec, f1, mcc]


def process_feature_set(feature_set_name, output_dir):
    print("Processing feature set: {}".format(feature_set_name))
    split_path = os.path.join("data_splits", feature_set_name)

    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data(split_path)
    print("Loaded data - Train: {}, Val: {}, Test: {}".format(
        len(X_train), len(X_val), len(X_test))
    )

    model = train_xgb_classifier(X_train, y_train, X_val, y_val)
    metrics = evaluate_model(model, X_test, y_test)

    print("-------- Metrics --------")
    print("     Accuracy : {:.4f}".format(metrics[0]))
    print("     Precision: {:.4f}".format(metrics[1]))
    print("     Recall   : {:.4f}".format(metrics[2]))
    print("     F1 Score : {:.4f}".format(metrics[3]))
    print("     MCC      : {:.4f}".format(metrics[4]))

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_csv = os.path.join(output_dir, "xgb_results.csv")
    save_results_to_csv(
        results_dict={feature_set_name: metrics},
        metric_names=["Accuracy", "Precision", "Recall", "F1-score", "MCC"],
        save_path=result_csv
    )

    # Generate SHAP plots and save
    shap_output = os.path.join("results", "shap", feature_set_name, "xgb")
    generate_and_save_shap_plots(model, X_train, X_test, shap_output)



def main(feature_set_name="basic"):
    print("Training XGBoost on  feature set {}...\n".format(feature_set_name))

    start_time = time.time()
    output_dir = os.path.join("results", feature_set_name)
    process_feature_set(feature_set_name, output_dir)
    end_time = time.time()
    print("XGBoost Model Takes {} seconds.".format(end_time-start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with specified feature set")
    parser.add_argument("--feature_set", choices=["basic", "cicflowmeter"], default="basic",
                        help="Specify feature set to use (default: basic)")
    args = parser.parse_args()
    main(args.feature_set)
