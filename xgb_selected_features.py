import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
from sklearn.metrics import matthews_corrcoef


def train_model(train_csv, val_csv, test_csv):

    # selected_features = ["Dur","TotPkts","TotBytes","SrcBytes","SrcRatio","PktRate","ByteRate","Proto","Dir"]
    selected_features = ["Bwd Pkt Len Max", "Fwd IAT Mean", "Tot Fwd Pkts", "Fwd IAT Min", "TotLen Fwd Pkts", "Bwd IAT Mean", "TotLen Bwd Pkts", "Bwd IAT Min", "Pkt Len Var"]
    # Load datasets
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)  # not used in this version, but loaded for possible future tuning
    test_df = pd.read_csv(test_csv)

    # Split features and labels
    X_train = train_df[selected_features]
    y_train = train_df["Label"]

    X_val = val_df[selected_features]
    y_val = val_df["Label"]

    X_test = test_df[selected_features]
    y_test = test_df["Label"]



    # # Initialize XGBoost classifier (simplified for older version)
    # model = xgb.XGBClassifier(
    #     n_estimators=100,
    #     max_depth=6,
    #     learning_rate=0.1
    # )

    # # Train model (no early stopping or eval_set)
    # model.fit(X_train, y_train)

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss"
    )

    # Train with early stopping using validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"MCC: {mcc:.4f}")

    print("Evaluation on Test Set:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Fast and compatible with SHAP 0.40+
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test.iloc[:100])  # Explanation object

    # Global feature importance
    shap.plots.beeswarm(shap_values)

    # Local explanation for first sample
    shap.plots.waterfall(shap_values[0])

    return model
# Execute model training
# model = train_model("Datasets/train_set.csv", "Datasets/val_set.csv", "Datasets/test_set.csv")
model = train_model("Datasets_test/train_set.csv", "Datasets_test/val_set.csv", "Datasets_test/test_set.csv")




