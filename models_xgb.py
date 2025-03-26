import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(train_csv, val_csv, test_csv):
    # Load datasets
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)  # not used in this version, but loaded for possible future tuning
    test_df = pd.read_csv(test_csv)

    # Split features and labels
    X_train, y_train = train_df.drop(columns=["Label"]), train_df["Label"]
    X_test, y_test = test_df.drop(columns=["Label"]), test_df["Label"]

    # Initialize XGBoost classifier (simplified for older version)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    # Train model (no early stopping or eval_set)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Evaluation on Test Set:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return model
# Execute model training
model = train_model("Datasets/train_set.csv", "Datasets/val_set.csv", "Datasets/test_set.csv")
