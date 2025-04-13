# ensured to run "brew install libomp" in terminal

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import os
from result_utils import save_results_to_csv
import time

def train_model(feature_set_name="basic"):
    start_time = time.time()
    # Load datasets
    base_path = os.path.join("data_splits", feature_set_name)
    train_csv = os.path.join(base_path, "train.csv")
    val_csv = os.path.join(base_path, "val.csv")
    test_csv = os.path.join(base_path, "test.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Sanitize column names to remove problematic characters
    train_df.columns = train_df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    val_df.columns = val_df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    test_df.columns = test_df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    print("\nFeature-label correlations:")
    print(train_df.corr(numeric_only=True)["Label"].sort_values(ascending=False))

    # Split features and labels
    X_train, y_train = train_df.drop(columns=["Label"]), train_df["Label"]
    X_test, y_test = test_df.drop(columns=["Label"]), test_df["Label"]

    best_model = lgb.LGBMClassifier(
        learning_rate=0.1,
        max_depth=-1,
        min_child_samples=20,
        n_estimators=300,
        subsample=0.6,
        random_state=42
    )
    best_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = best_model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("\nEvaluation on Test Set:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"Test Error: {1 - acc:.4f}")


    # Save results
    metrics = [acc, precision, recall, f1, mcc]
    result_dir = os.path.join("results", feature_set_name)
    os.makedirs(result_dir, exist_ok=True)
    result_csv = os.path.join(result_dir, "lgbm_results.csv")
    save_results_to_csv(
        results_dict={feature_set_name: metrics},
        metric_names=["Accuracy", "Precision", "Recall", "F1-score", "MCC"],
        save_path=result_csv
    )
    process_time = time.time() - start_time
    print("LightGBM Model Takes {} seconds.".format(process_time))

    return best_model



def main(feature_set_name="basic"):

    # Execute model training with LightGBM
    # Execute model training
    train_model(feature_set_name)



# Evaluation on Test Set:
# Accuracy : 0.9625
# Precision: 0.9626
# Recall   : 0.9701
# F1 Score : 0.9664
# Matthews Correlation Coefficient: 0.9241
# Test Error: 0.0375

if __name__ == "__main__":
    main()
