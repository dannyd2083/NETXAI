
# Note:
# This is the training procedure to find out
# the best suitable parameters to build random forest model
# This took very long time (more than a few hours) to run.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(train_csv, val_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)  # Currently unused
    test_df = pd.read_csv(test_csv)

    X_train, y_train = train_df.drop(columns=["Label"]), train_df["Label"]
    X_test, y_test = test_df.drop(columns=["Label"]), test_df["Label"]
    return X_train, y_train, X_test, y_test


def evaluate_param_with_cv(X, y, param_name, param_values, fixed_params=None, n_splits=5, random_state=42):
    if fixed_params is None:
        fixed_params = {}

    results = {}
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for val in param_values:
        fold_errors = []

        for train_idx, val_idx in kf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            params = {param_name: val, **fixed_params}
            model = RandomForestClassifier(random_state=random_state, **params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            f1 = f1_score(y_val, y_pred, zero_division=0)
            error = 1 - f1
            fold_errors.append(error)

        avg_error = np.mean(fold_errors)
        results[val] = avg_error
        print(f"{param_name} = {str(val):<7} → Avg CV F1 Error: {avg_error:.4f}")

    return results


def plot_cv_results(results, param_name):
    keys = list(results.keys())
    errors = [results[k] for k in keys]
    labels = [str(k) for k in keys]

    plt.figure(figsize=(8, 5))
    plt.plot(labels, errors, marker='o', linestyle='-', color='teal')
    plt.title(f"5-Fold CV F1 Error vs {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("F1 Error (1 - F1 Score)")
    plt.grid(True)

    best_idx = np.argmin(errors)
    plt.scatter(labels[best_idx], errors[best_idx], color='red', zorder=5)
    plt.text(labels[best_idx], errors[best_idx] + 0.002, f"Best: {errors[best_idx]:.4f}", ha='center', color='red')
    plt.tight_layout()
    plt.show()


def find_best_param(train_csv, val_csv, test_csv, param_name, param_values, fixed_params=None, random_state=42):
    X_train, y_train, X_test, y_test = load_data(train_csv, val_csv, test_csv)
    results = evaluate_param_with_cv(X_train, y_train, param_name, param_values, fixed_params, random_state=random_state)
    plot_cv_results(results, param_name)
    best_val = min(results, key=results.get)
    return best_val, results

best_max_features, max_feat_results = find_best_param(
    train_csv="Datasets/train_set.csv",
    val_csv="Datasets/val_set.csv",
    test_csv="Datasets/test_set.csv",
    param_name="max_features",
    param_values=['sqrt', 'log2', None, 2, 4, 6, 8, 10],
    fixed_params={},  # No other fixed params yet
    random_state=42
)

best_n_estimators, n_est_results = find_best_param(
    train_csv="Datasets/train_set.csv",
    val_csv="Datasets/val_set.csv",
    test_csv="Datasets/test_set.csv",
    param_name="n_estimators",
    param_values=[10, 50, 100, 150, 200],
    fixed_params={"max_features": best_max_features},
    random_state=42
)

best_max_depth, max_depth_results = find_best_param(
    train_csv="Datasets/train_set.csv",
    val_csv="Datasets/val_set.csv",
    test_csv="Datasets/test_set.csv",
    param_name="max_depth",
    param_values=[None, 2, 5, 10, 20, 30],
    fixed_params={
        "max_features": best_max_features,
        "n_estimators": best_n_estimators
    },
    random_state=42
)

"""
The best features for new data found are the following:
max_features = 6       → Avg CV F1 Error: 0.0273
n_estimators = 150     → Avg CV F1 Error: 0.0272
max_depth = 20      → Avg CV F1 Error: 0.0263
"""
