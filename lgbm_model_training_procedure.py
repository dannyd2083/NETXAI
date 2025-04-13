# ensured to run "brew install libomp" in terminal

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb
from itertools import product

def load_data(train_csv, val_csv, test_csv, label_col='Label', verbose=True):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_df.columns = train_df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    val_df.columns = val_df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    test_df.columns = test_df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    X_train, y_train = train_df.drop(columns=[label_col]), train_df[label_col]
    X_val, y_val = val_df.drop(columns=[label_col]), val_df[label_col]
    X_test, y_test = test_df.drop(columns=[label_col]), test_df[label_col]

    if verbose:
        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
        print("\nTop correlated features with label (train set):")
        print(train_df.corr(numeric_only=True)[label_col].sort_values(ascending=False).head(10))

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_grid_with_cv(X, y, param_grid, n_splits=5, random_state=42):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    param_combinations = list(product(*param_grid.values()))
    param_keys = list(param_grid.keys())

    results = []

    for values in param_combinations:
        params = dict(zip(param_keys, values))
        fold_errors = []

        for train_idx, val_idx in kf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(random_state=random_state, **params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            f1 = f1_score(y_val, y_pred, zero_division=0)
            error = 1 - f1
            fold_errors.append(error)

        avg_error = np.mean(fold_errors)
        print(f"{params} → Avg CV F1 Error: {avg_error:.4f}")
        results.append((params, avg_error))

    results.sort(key=lambda x: x[1])
    best_params, best_error = results[0]
    print(" Best Params:", best_params)
    print(f"Best Avg F1 Error: {best_error:.4f}")
    return best_params, results

if __name__ == "__main__":
    # Load all datasets
    X_train, y_train, _, _, X_test, y_test = load_data(
        "Datasets/train_set.csv",
        "Datasets/val_set.csv",
        "Datasets/test_set.csv"
    )

    # Define grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_samples': [10, 20, 50],
        'subsample': [0.6, 0.8, 1.0]
    }

    # Run grid search with 5-fold CV
    best_params, all_results = evaluate_grid_with_cv(X_train, y_train, param_grid)

    # Retrain on full training set using best params and evaluate on test
    print("\n Retraining best model on full training set and evaluating on test set...")
    final_model = lgb.LGBMClassifier(random_state=42, **best_params)
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)

    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"\n Test Set Evaluation:")
    print(f"F1 Score: {test_f1:.4f}")

# Best Params: {'n_estimators': 300, 'max_depth': -1, 'learning_rate': 0.1, 'min_child_samples': 20, 'subsample': 0.6}
# Best Avg F1 Error: 0.0337

