import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time


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


def train_logistic_regression(X_train, y_train, params):
    """
    Train Logistic Regression model with optimal parameters
    """
    print(f"Training Logistic Regression model with parameters: {params}")
    start_time = time.time()

    lr = LogisticRegression(**params, random_state=42)
    lr.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")

    return lr


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set
    """
    print("Evaluating model on test set...")
    start_time = time.time()

    # Predict on test set
    y_pred = model.predict(X_test)

    prediction_time = time.time() - start_time

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print("\nEvaluation on Test Set:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Malicious'],
                yticklabels=['Normal', 'Malicious'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Save confusion matrix
    plt.savefig("results/logistic_regression_confusion_matrix.png")

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [acc, precision, recall, f1]
    })
    metrics_df.to_csv("results/logistic_regression_metrics.csv", index=False)

    return acc, precision, recall, f1


def analyze_feature_importance(model, feature_names):
    """
    Analyze feature importance based on model coefficients
    """
    print("\nAnalyzing feature importance...")

    # Get coefficients
    coefficients = model.coef_[0]

    # Create DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Absolute_Importance': np.abs(coefficients)
    })

    # Sort by absolute importance
    importance_df = importance_df.sort_values('Absolute_Importance', ascending=False)

    # Plot feature importance (top 20 features)
    plt.figure(figsize=(12, 10))
    top_features = importance_df.head(20)
    colors = ['red' if c < 0 else 'blue' for c in top_features['Coefficient']]

    bars = plt.barh(top_features['Feature'], top_features['Absolute_Importance'], color=colors)
    plt.title('Top 20 Feature Importance for Logistic Regression Model')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.savefig("results/logistic_regression_feature_importance.png")

    # Save feature importance to CSV
    importance_df.to_csv("results/logistic_regression_feature_importance.csv", index=False)

    print("Top 10 most important features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Coefficient']:.4f}")

    return importance_df


def visualize_decision_boundary(model, X_test, y_test):
    """
    Create a simple visualization of the decision boundary
    using dimensionality reduction to 2D
    """
    try:
        from sklearn.decomposition import PCA

        print("\nVisualizing decision boundary...")

        # Reduce dimensionality to 2D for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)

        # Get decision function values (distance from decision boundary)
        decision_values = model.decision_function(X_test)

        # Create plot
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test,
                         cmap=plt.cm.coolwarm, alpha=0.8,
                         edgecolors='k', s=40)

        plt.colorbar(sc, label='Class')
        plt.title('PCA Projection with Decision Boundary')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Save visualization
        plt.savefig("results/logistic_regression_decision_boundary.png")

    except Exception as e:
        print(f"Could not create decision boundary visualization: {str(e)}")


def main():
    """
    Main function to train and evaluate Logistic Regression model
    """
    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Load datasets
    train_df, val_df, test_df = load_datasets(
        "Datasets/train_set.csv",
        "Datasets/val_set.csv",
        "Datasets/test_set.csv"
    )

    # Prepare data with handling for missing values
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_data(train_df, val_df, test_df)

    # Optimize parameters
    best_params = optimize_parameters(X_train, y_train, X_val, y_val)

    # Train model
    lr_model = train_logistic_regression(X_train, y_train, best_params)

    # Evaluate model
    acc, precision, recall, f1 = evaluate_model(lr_model, X_test, y_test)

    # Analyze feature importance
    analyze_feature_importance(lr_model, feature_names)

    # Visualize decision boundary
    visualize_decision_boundary(lr_model, X_test, y_test)

    print("\nLogistic Regression model training and evaluation completed!")
    return lr_model


if __name__ == "__main__":
    model = main()