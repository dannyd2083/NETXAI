# ensured to run "brew install libomp" in terminal

import pandas as pd
import lightgbm as lgb
#from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_model(train_csv, val_csv, test_csv):
    # Load datasets
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)  # Optional for future use
    test_df = pd.read_csv(test_csv)

    print("\nFeature-label correlations:")
    print(train_df.corr(numeric_only=True)["Label"].sort_values(ascending=False))

    # Split features and labels
    X_train, y_train = train_df.drop(columns=["Label"]), train_df["Label"]
    X_test, y_test = test_df.drop(columns=["Label"]), test_df["Label"]

    # Comment the following section
    # because it took more than an hour to run
    # The result is shown after the code:

    # Define hyperparameter grid for LightGBM
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_samples': [10, 20, 50],
        'subsample': [0.6, 0.8, 1.0]
    }

    # Best Hyperparameters Found:
    # {'learning_rate': 0.1, 'max_depth': -1, 'min_child_samples': 10, 'n_estimators': 300, 'subsample': 0.6}
    # Best Cross-Validation F1 Score: 0.9663
    """
    Evaluation on Test Set:
    Accuracy : 0.9627
    Precision: 0.9632
    Recall   : 0.9698
    F1 Score : 0.9665
    Test Error: 0.0373
    """

    grid_search = GridSearchCV(
        estimator=lgb.LGBMClassifier(random_state=42),
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\nBest Hyperparameters Found:")
    print(grid_search.best_params_)
    print(f"Best Cross-Validation F1 Score: {grid_search.best_score_:.4f}")

    # Predict on test set using best model
    y_pred = best_model.predict(X_test)

    # Evaluate test performance
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nEvaluation on Test Set:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Test Error: {1 - acc:.4f}")

    return best_model


# Execute model training with LightGBM
model = train_model("Datasets/train_set.csv", "Datasets/val_set.csv", "Datasets/test_set.csv")
