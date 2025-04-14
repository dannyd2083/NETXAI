## To find the best params for logistic regression model
## For the same data only run once, because it takes a long time

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
import os



def load_datasets(train_csv, val_csv, test_csv):
    """
    Load training, validation and testing datasets
    """
    print("Loading datasets...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    return train_df, val_df, test_df


def prepare_data(train_df, val_df, test_df):
    """
    Prepare features and labels for Logistic Regression model, handling missing values
    """
    # Split features and labels
    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]

    X_val = val_df.drop(columns=["Label"])
    y_val = val_df["Label"]

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]

    # Handle missing values with imputation
    print("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    # Replace any remaining NaNs with 0
    X_train_imputed = np.nan_to_num(X_train_imputed)
    X_val_imputed = np.nan_to_num(X_val_imputed)
    X_test_imputed = np.nan_to_num(X_test_imputed)

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    print("Data preparation complete.")
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, X_train.columns.tolist()


def optimize_parameters(X_train, y_train, X_val, y_val):
    """
    Find optimal parameters for Logistic Regression using validation set
    Only use compatible solver/penalty combinations
    """
    print("Finding optimal parameters...")

    # Define valid parameter combinations
    param_combinations = [
        {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 10.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 100.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000},

        {'C': 0.001, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000},
        {'C': 0.01, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000},
        {'C': 0.1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000},
        {'C': 1.0, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000},
        {'C': 10.0, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000},
        {'C': 100.0, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000},

        {'C': 0.001, 'penalty': 'l2', 'solver': 'saga', 'max_iter': 1000},
        {'C': 0.01, 'penalty': 'l2', 'solver': 'saga', 'max_iter': 1000},
        {'C': 0.1, 'penalty': 'l2', 'solver': 'saga', 'max_iter': 1000},
        {'C': 1.0, 'penalty': 'l2', 'solver': 'saga', 'max_iter': 1000},
        {'C': 10.0, 'penalty': 'l2', 'solver': 'saga', 'max_iter': 1000},
        {'C': 100.0, 'penalty': 'l2', 'solver': 'saga', 'max_iter': 1000},

        {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 10.0, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 100.0, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000},

        {'C': 0.001, 'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 10.0, 'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 1000},
        {'C': 100.0, 'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 1000},
    ]

    best_score = 0
    best_params = None

    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations...")

    for i, params in enumerate(param_combinations):
        try:
            # Train model with current parameters
            lr = LogisticRegression(**params, random_state=42)
            lr.fit(X_train, y_train)

            # Predict on validation set
            y_pred = lr.predict(X_val)
            f1 = f1_score(y_val, y_pred)

            print(f"Params: {params}, F1 Score: {f1:.4f}")

            if f1 > best_score:
                best_score = f1
                best_params = params

            # Print progress every 5 combinations
            if (i + 1) % 5 == 0 or i + 1 == total_combinations:
                print(f"Progress: {i + 1}/{total_combinations} combinations tested")

        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            continue

    if best_params:
        print(f"Best parameters: {best_params}")
        print(f"Best F1 score on validation set: {best_score:.4f}")
    else:
        print("No valid parameter combination found. Using default parameters.")
        best_params = {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000}

    return best_params


def main(feature_set_name="basic"):
    """
    Main function to train and evaluate Logistic Regression model
    """
    print("--------- Training Logistic Regression on {} feature set -----------".format(feature_set_name))
    # Create results directory
    os.makedirs("results", exist_ok=True)
    base_path = os.path.join("data_splits", feature_set_name)
    # Load datasets
    train_df, val_df, test_df = load_datasets(
        os.path.join(base_path, "train.csv"),
        os.path.join(base_path, "val.csv"),
        os.path.join(base_path, "test.csv")
    )

    # Prepare data with handling for missing values
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_data(train_df, val_df, test_df)

    # Optimize parameters
    best_params = optimize_parameters(X_train, y_train, X_val, y_val)
    return best_params