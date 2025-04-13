# ensured to run "brew install libomp" in terminal

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def train_model(train_csv, val_csv, test_csv):
    # Load datasets
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

    return best_model

# Execute model training with LightGBM
model = train_model("Datasets/train_set.csv", "Datasets/val_set.csv", "Datasets/test_set.csv")

# Evaluation on Test Set:
# Accuracy : 0.9625
# Precision: 0.9626
# Recall   : 0.9701
# F1 Score : 0.9664
# Matthews Correlation Coefficient: 0.9241
# Test Error: 0.0375
