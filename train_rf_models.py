# Note:
# This took relatively reasonable time to run.
# This contains only the final deducted random forest model.
# This doesn't contain the training procedure
# for choosing the best parameters for this random forest model
# If you want to see that,
# please refer to random_forest_training_procedure.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score
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
    val_df = pd.read_csv(val_csv)  # Not used currently
    test_df = pd.read_csv(test_csv)

    # Split features and labels
    X_train, y_train = train_df.drop(columns=["Label"]), train_df["Label"]
    X_test, y_test = test_df.drop(columns=["Label"]), test_df["Label"]

    # # Initialize Random Forest classifier
    # model = RandomForestClassifier(
    #     n_estimators=150,
    #     max_depth=6,
    #     max_features=20,
    #     random_state=42
    # )
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        max_features=20,
        random_state=42,
        n_jobs=-1  # 
    )


    # n-fold cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
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
    mcc = matthews_corrcoef(y_test, y_pred)

    print("Evaluation on Test Set:")
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
    result_csv = os.path.join(result_dir, "rf_results.csv")
    save_results_to_csv(
        results_dict={feature_set_name: metrics},
        metric_names=["Accuracy", "Precision", "Recall", "F1-score", "MCC"],
        save_path=result_csv
    )

    end_time = time.time()
    print("Random Forest Model Takes {} seconds.".format(end_time-start_time))

    return model


def main(feature_set_name="basic"):
    # Execute model training
    model = train_model(feature_set_name)


if __name__ == "__main__":
    main()
