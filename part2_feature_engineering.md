# Install necessary packages

```python
%pip install -r requirements.txt
```
# Part 2: Time Series Features & Tree-Based Models

**Objective:** Extract basic time-series features from heart rate data, train Random Forest and XGBoost models, and compare their performance.

## 1. Setup

Import necessary libraries.

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
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
        DataFrame containing the data with timestamp parsed as datetime
    """
    # YOUR CODE HERE
    # Load the CSV file using pandas
    # Make sure to parse the timestamp column as datetime
    df = pd.read_csv(file_path, parse_dates = ["timestamp"])

    return df
```

## 3. Feature Engineering

Implement `extract_rolling_features` to calculate rolling mean and standard deviation for the `heart_rate`.

```python
def extract_rolling_features(df, window_size_seconds):
    """
    Calculate rolling mean and standard deviation for heart rate.
    
    Args:
        df: DataFrame with timestamp and heart_rate columns
        window_size_seconds: Size of the rolling window in seconds
        
    Returns:
        DataFrame with added hr_rolling_mean and hr_rolling_std columns
    """
    # YOUR CODE HERE
    # 1. Sort data by timestamp
    df_sorted = df.sort_values('timestamp')
    
    # 2. Set timestamp as index (this allows time-based operations)
    df_indexed = df_sorted.set_index('timestamp')
    
    # 3. Calculate rolling mean and standard deviation
    # First, create a rolling window object based on time:
    rolling_window = df_indexed['heart_rate'].rolling(window = f'{window_size_seconds}s')

    # Then calculate statistics on this window:
    hr_mean = rolling_window.mean()
    hr_std = rolling_window.std()
    
    # 4. Add the new columns back to the dataframe
    df_indexed['hr_rolling_mean'] = hr_mean
    df_indexed['hr_rolling_std'] = hr_std
    
    # 5. Reset index to bring timestamp back as a column
    df_result = df_indexed.reset_index()
    
    # 6. Handle any NaN values (rolling calculations create NaNs at the beginning)
    # You can use fillna, dropna, or other methods depending on your strategy
    df_result['hr_rolling_mean'] = df_result['hr_rolling_mean'].bfill().ffill()
    df_result['hr_rolling_std'] = df_result['hr_rolling_std'].bfill().ffill()
    
    return df_result.copy()
```

## 4. Data Preparation

Implement `prepare_data_part2` using the newly engineered features.

```python
def prepare_data_part2(df_with_features, test_size=0.2, random_state=42):
    """
    Prepare data for modeling with time-series features.
    
    Args:
        df_with_features: DataFrame with original and rolling features
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # YOUR CODE HERE
    # 1. Select relevant features including the rolling features
    X = df_with_features[["age", "systolic_bp", "diastolic_bp", "glucose_level", "bmi", "hr_rolling_mean", "hr_rolling_std"]]
    
    # 2. Select target variable (disease_outcome)
    y = df_with_features[["disease_outcome"]]

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = test_size,
                                                        random_state = random_state,
                                                        stratify = y)
    
    # 4. Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train =pd.DataFrame(imputer.fit_transform(X_train), columns = X.columns, index = X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns = X.columns, index = X_test.index)

    return X_train, X_test, y_train, y_test
```

## 5. Random Forest Model

Implement `train_random_forest`.

```python
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        random_state: Random seed for reproducibility
        
    Returns:
        Trained Random Forest model
    """
    # YOUR CODE HERE
    # Initialize and train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators = n_estimators,
                                   max_depth = max_depth,
                                   random_state = random_state).fit(X_train, y_train)
    
    return model
```

## 6. XGBoost Model

Implement `train_xgboost`.

```python
def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
    """
    Train an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting rounds
        learning_rate: Boosting learning rate
        max_depth: Maximum depth of a tree
        random_state: Random seed for reproducibility
        
    Returns:
        Trained XGBoost model
    """
    # YOUR CODE HERE
    # Initialize and train an XGBClassifier
    model = xgb.XGBClassifier(n_estimators = n_estimators,
                            learning_rate = learning_rate,
                            max_depth = max_depth,
                            random_state = random_state).fit(X_train, y_train)

    return model
```

## 7. Model Comparison

Calculate and compare AUC scores for both models.

```python
# YOUR CODE HERE
def calculate_auc(rf_model, xgb_model, X_test, y_test):
    # 1. Generate probability predictions fo
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # 2. Calculate AUC scores
    rf_auc = roc_auc_score(y_test, rf_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    
    # 3. Compare the performance
    print(f"Random Forest AUC: {rf_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    
    return {"rf_auc": rf_auc, "xgb_auc": xgb_auc}
```

## 8. Save Results

Save the AUC scores to a text file.

```python
def save_results(metrics):
    # 1. Create 'results' directory if it doesn't exist
    os.makedirs("results", exist_ok = True)

    # 2. Format AUC scores as strings
    result_str = ""
    for metric, value in metrics.items():
        result_str += f"{metric}: {value:.4f}\n"

    # 3. Write scores to 'results/results_part2.txt'
    with open("results/results_part2.txt", "w") as f:
        f.write(result_str)
```

## 9. Main Execution

Run the complete workflow.

```python
# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Extract rolling features
    window_size = 300  # 5 minutes in seconds
    df_with_features = extract_rolling_features(df, window_size)
    
    # 3. Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part2(df_with_features)
    
    # 4. Train models
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # 5. Calculate AUC scores
    metrics = calculate_auc(rf_model, xgb_model, X_test, y_test)
    
    # 6. Save results
    save_results(metrics)
```