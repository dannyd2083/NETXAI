import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
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
    Prepare features and labels for KNN model, handling missing values
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

    # Check if there are any remaining NaN values
    if np.isnan(X_train_imputed).any():
        print("Warning: There are still NaN values after imputation in the training set.")
        # Replace any remaining NaNs with 0
        X_train_imputed = np.nan_to_num(X_train_imputed)

    if np.isnan(X_val_imputed).any():
        print("Warning: There are still NaN values after imputation in the validation set.")
        X_val_imputed = np.nan_to_num(X_val_imputed)

    if np.isnan(X_test_imputed).any():
        print("Warning: There are still NaN values after imputation in the test set.")
        X_test_imputed = np.nan_to_num(X_test_imputed)

    # Scale features (important for KNN)
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Final check for any NaN values
    print("Checking for any remaining NaN values...")
    if np.isnan(X_train_scaled).any():
        print("Warning: NaN values found in scaled training data. Replacing with zeros.")
        X_train_scaled = np.nan_to_num(X_train_scaled)

    if np.isnan(X_val_scaled).any():
        print("Warning: NaN values found in scaled validation data. Replacing with zeros.")
        X_val_scaled = np.nan_to_num(X_val_scaled)

    if np.isnan(X_test_scaled).any():
        print("Warning: NaN values found in scaled test data. Replacing with zeros.")
        X_test_scaled = np.nan_to_num(X_test_scaled)

    print("Data preparation complete.")
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test


def find_best_k(X_train, y_train, X_val, y_val, k_range=range(1, 21)):
    """
    Find the optimal K value using validation set
    """
    print("Finding optimal K value...")
    k_scores = []

    for k in k_range:
        # Initialize and train KNN model
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Predict on validation set
        y_pred = knn.predict(X_val)

        # Calculate F1 score (balanced metric for potentially imbalanced classes)
        f1 = f1_score(y_val, y_pred)
        k_scores.append(f1)

        print(f"K={k}, F1 Score={f1:.4f}")

    # Find best K
    best_k = k_range[np.argmax(k_scores)]
    best_score = max(k_scores)
    print(f"Best K value: {best_k} with F1 Score: {best_score:.4f}")

    # Plot K values vs F1 scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, k_scores, marker='o')
    plt.title('F1 Score for Different K Values')
    plt.xlabel('K Value (Number of Neighbors)')
    plt.ylabel('F1 Score')
    plt.grid(True)

    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/knn_k_optimization.png")

    return best_k


def train_knn_model(X_train, y_train, k):
    """
    Train KNN model with the optimal K value
    """
    print(f"Training KNN model with K={k}...")
    start_time = time.time()

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")

    return knn


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

    # Save confusion matrix
    plt.savefig("results/knn_confusion_matrix.png")

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [acc, precision, recall, f1]
    })
    metrics_df.to_csv("results/knn_metrics.csv", index=False)

    return acc, precision, recall, f1


def analyze_feature_importance(model, X_train, y_train, feature_names):
    """
    Analyze feature importance for KNN by removing features one by one
    """
    print("\nAnalyzing feature importance...")

    # Initialize baseline model
    knn_baseline = KNeighborsClassifier(n_neighbors=model.n_neighbors)
    knn_baseline.fit(X_train, y_train)
    baseline_score = knn_baseline.score(X_train, y_train)

    importance_scores = []

    # Test model performance with each feature removed
    for i in range(X_train.shape[1]):
        # Create copy of data without one feature
        X_train_reduced = np.delete(X_train, i, axis=1)

        # Train model on reduced feature set
        knn_reduced = KNeighborsClassifier(n_neighbors=model.n_neighbors)
        knn_reduced.fit(X_train_reduced, y_train)

        # Score model
        reduced_score = knn_reduced.score(X_train_reduced, y_train)

        # Calculate importance (how much accuracy drops when feature is removed)
        importance = baseline_score - reduced_score
        importance_scores.append(importance)

    # Create DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })

    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance for KNN Model')
    plt.tight_layout()
    plt.savefig("results/knn_feature_importance.png")

    # Save feature importance to CSV
    importance_df.to_csv("results/knn_feature_importance.csv", index=False)

    print("Top 5 most important features:")
    print(importance_df.head(5))

    return importance_df


def main():
    """
    Main function to train and evaluate KNN model
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
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(train_df, val_df, test_df)

    # Find optimal K value
    best_k = find_best_k(X_train, y_train, X_val, y_val)

    # Train KNN model
    knn_model = train_knn_model(X_train, y_train, best_k)

    # Evaluate model
    acc, precision, recall, f1 = evaluate_model(knn_model, X_test, y_test)

    # Get feature names (column names without the Label column)
    feature_names = train_df.drop(columns=["Label"]).columns.tolist()

    # Analyze feature importance
    analyze_feature_importance(knn_model, X_train, y_train, feature_names)

    print("\nKNN model training and evaluation completed!")
    return knn_model


if __name__ == "__main__":
    model = main()