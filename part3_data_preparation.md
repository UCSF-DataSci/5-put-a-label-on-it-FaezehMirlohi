# Install necessary packages

```python
%pip install -r requirements.txt
```
# Part 3: Practical Data Preparation

**Objective:** Handle categorical features using One-Hot Encoding and address class imbalance using SMOTE.

## 1. Setup

Import necessary libraries.

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
```

## 2. Data Loading

Load the dataset.

```python
def load_data(file_path):
    """
    Load the synthetic health data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    # YOUR CODE HERE
    # Load the CSV file using pandas
    df = pd.read_csv(file_path)

    return df
```

## 3. Categorical Feature Encoding

Implement `encode_categorical_features` using `OneHotEncoder`.

```python
def encode_categorical_features(df, column_to_encode='smoker_status'):
    """
    Encode a categorical column using OneHotEncoder.
    
    Args:
        df: Input DataFrame
        column_to_encode: Name of the categorical column to encode
        
    Returns:
        DataFrame with the categorical column replaced by one-hot encoded columns
    """
    # YOUR CODE HERE
    # 1. Extract the categorical column
    cat_var = df[[column_to_encode]]

    # 2. Apply OneHotEncoder
    encoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
    encoded = encoder.fit_transform(cat_var)

    # 3. Create new column names
    encoded_cols = encoder.get_feature_names_out([column_to_encode])

    # 4. Replace the original categorical column with the encoded columns
    encoded_df = pd.DataFrame(encoded, columns = encoded_cols, index = df.index)

    df = df.drop(columns = [column_to_encode])
    df = pd.concat([df, encoded_df], axis = 1)
    
    # Placeholder return - replace with your implementation
    return df.copy()
```

## 4. Data Preparation

Implement `prepare_data_part3` to handle the train/test split correctly.

```python
def prepare_data_part3(df, test_size=0.2, random_state=42):
    """
    Prepare data with categorical encoding.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # YOUR CODE HERE
    # 1. Encode categorical features using the encode_categorical_features function
    df = encode_categorical_features(df)

    # 2. Select relevant features (including the one-hot encoded ones) and the target
    X = df.drop(columns = ["patient_id", "timestamp", "disease_outcome"])
    y = df["disease_outcome"]

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = test_size,
                                                        random_state = random_state,
                                                        stratify = y)
    
    imputer = SimpleImputer(strategy='mean')
    X_train =pd.DataFrame(imputer.fit_transform(X_train), columns = X.columns, index = X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns = X.columns, index = X_test.index)

    # encoder
    encoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')

    # Placeholder return - replace with your implementation
    return X_train, X_test, y_train, y_test, encoder
```

## 5. Handling Imbalanced Data

Implement `apply_smote` to oversample the minority class.

```python
def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to oversample the minority class.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        
    Returns:
        Resampled X_train and y_train with balanced classes
    """
    # YOUR CODE HERE
    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state = random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Return resampled data
    return X_resampled, y_resampled
```

## 6. Model Training and Evaluation

Train a model on the SMOTE-resampled data and evaluate it.

```python
def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained logistic regression model
    """
    # YOUR CODE HERE
    # Initialize and train a LogisticRegression model
    model = LogisticRegression(max_iter = 1000).fit(X_train, y_train)
    
    return model

def calculate_evaluation_metrics(model, X_test, y_test):
    """
    Calculate classification evaluation metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, auc, and confusion_matrix
    """
    # YOUR CODE HERE
    # 1. Generate predictions
    y_pred = model.predict(X_test)

    # 2. Calculate metrics: accuracy, precision, recall, f1, auc
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # 3. Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # 4. Return metrics in a dictionary
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc, "confusion_matrix": cm}
```

## 7. Save Results

Save the evaluation metrics to a text file.

```python
# YOUR CODE HERE
def save_results(metrics):
    # 1. Create 'results' directory if it doesn't exist
    os.makedirs("results", exist_ok = True)

    # 2. Format metrics as strings
    result_str = ""
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            result_str += f"{metric}: {value:.4f}\n"

    # 3. Write metrics to 'results/results_part3.txt'
    with open("results/results_part3.txt", "w") as f:
        f.write(result_str)
```

## 8. Main Execution

Run the complete workflow.

```python
# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Prepare data with categorical encoding
    X_train, X_test, y_train, y_test, encoder = prepare_data_part3(df)
    
    # 3. Apply SMOTE to balance the training data
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # 4. Train model on resampled data
    model = train_logistic_regression(X_train_resampled, y_train_resampled)
    
    # 5. Evaluate on original test set
    metrics = calculate_evaluation_metrics(model, X_test, y_test)
    
    # 6. Print metrics
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    # 7. Save results
    save_results(metrics)
    
    # 8. Load Part 1 results for comparison
    part1_metrics = {}
    try:
        with open('results/results_part1.txt', 'r') as f:
             for line in f:
                try:
                    key, value = line.strip().split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key != "confusion_matrix":
                        part1_metrics[key] = float(value)
                except ValueError:
                    continue
        
        # 9. Compare models
        comparison = compare_models(part1_metrics, metrics)
        print("\nModel Comparison (improvement percentages):")
        for metric, improvement in comparison.items():
            print(f"{metric}: {improvement:.2f}%")
    except FileNotFoundError:
        print("Part 1 results not found. Run part1_introduction.ipynb first.")
```

## 9. Compare Results

Implement a function to compare model performance between balanced and imbalanced data.

```python
def compare_models(part1_metrics, part3_metrics):
    """
    Calculate percentage improvement between models trained on imbalanced vs. balanced data.
    
    Args:
        part1_metrics: Dictionary containing evaluation metrics from Part 1 (imbalanced)
        part3_metrics: Dictionary containing evaluation metrics from Part 3 (balanced)
        
    Returns:
        Dictionary with metric names as keys and improvement percentages as values
    """
    # YOUR CODE HERE
    improvements = {}

    # 1. Calculate percentage improvement for each metric
    for metric, value in part1_metrics.items():
        if metric != "confusion_matrix":
            improvement = ((part3_metrics[metric] - value) / part1_metrics[metric]) * 100
            improvements[metric] = round(improvement, 2)

    # 2. Handle metrics where higher is better (most metrics) and where lower is better
    # 3. Return a dictionary with metric names and improvement percentages
    return improvements
    ```