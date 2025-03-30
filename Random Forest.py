import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score


def train_model(train_csv, val_csv, test_csv):
    # Load datasets
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)  # Not used currently
    test_df = pd.read_csv(test_csv)

    print(train_df.corr(numeric_only=True)["Label"].sort_values(ascending=False))

    # Split features and labels
    X_train, y_train = train_df.drop(columns=["Label"]), train_df["Label"]
    X_test, y_test = test_df.drop(columns=["Label"]), test_df["Label"]

    # Initialize Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    # 5-fold cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_error = 1 - cv_scores.mean()
    print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f}")
    print(f"5-Fold Cross-Validation Error   : {cv_error:.4f}")

    # Train model
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
    print(f"Test Error: {1 - acc:.4f}")
    return model


# Execute model training
model = train_model("Datasets/train_set.csv", "Datasets/val_set.csv", "Datasets/test_set.csv")
