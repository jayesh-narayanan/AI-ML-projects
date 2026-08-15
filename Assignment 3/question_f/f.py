import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import importlib
from itertools import product
import sys
file_path = r'C:\IITD\Sem-5\COL774\Assignment 3\decision_tree.py'

# Load the module
spec = importlib.util.spec_from_file_location("decision_tree", file_path)
decision_tree = importlib.util.module_from_spec(spec)
sys.modules["decision_tree"] = decision_tree
spec.loader.exec_module(decision_tree)
# Use the class
DecisionTree= decision_tree.DecisionTree
from decision_tree import DecisionTree

# LOAD DATA 

print("Loading datasets...")
train_df = pd.read_csv(r'C:\IITD\Sem-5\COL774\Assignment 3\train.csv')
test_df = pd.read_csv(r'C:\IITD\Sem-5\COL774\Assignment 3\test.csv')
val_df = pd.read_csv(r'C:\IITD\Sem-5\COL774\Assignment 3\validation.csv')
temp_tree = DecisionTree()
temp_tree.feature_types(train_df)
categorical_cols = [col for col, vals in temp_tree.categorical_info.items() if len(vals) > 2]

# Perform one-hot encoding first
for col in categorical_cols:
    unique_values = temp_tree.categorical_info[col]
    for val in unique_values:
        new_col = f"{col}_{val}"
        train_df[new_col] = (train_df[col] == val).astype(int)
        test_df[new_col] = (test_df[col] == val).astype(int)
        val_df[new_col]=(val_df[col] == val).astype(int)
    train_df.drop(columns=[col], inplace=True)
    test_df.drop(columns=[col], inplace=True)
    val_df.drop(columns=[col], inplace=True)

# Separate features and labels
X_train = train_df.drop('result', axis=1)
y_train = train_df['result']

X_test = test_df.drop('result', axis=1)
y_test = test_df['result']

X_val = val_df.drop('result', axis=1)
y_val = val_df['result']

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Validation set: {len(X_val)} samples")
print()

# initialize the range of tunable params
n_estimators_range = [50, 150, 250, 350]  
max_features_range = [0.1, 0.3, 0.5, 0.7, 0.9]  
min_samples_split_range = [2, 4, 6, 8, 10]  
print("RANDOM FOREST GRID SEARCH")
print(f"Total combinations: {len(n_estimators_range) * len(max_features_range) * len(min_samples_split_range)}")
print()

# === GRID SEARCH WITH 3 NESTED LOOPS ====

best_val_acc = 0
best_params = {}
best_model = None
best_test_predictions = None

iteration = 0
total_iterations = len(n_estimators_range) * len(max_features_range) * len(min_samples_split_range)

print("Starting grid search...")
print()

for n_estimators in n_estimators_range:
    for max_features in max_features_range:
        for min_samples_split in min_samples_split_range:
            iteration += 1
            
            print(f"[{iteration}/{total_iterations}] Training with:")
            print(f"  n_estimators={n_estimators}, max_features={max_features}, min_samples_split={min_samples_split}")
            
            # Create Random Forest with entropy criterion
            rf = RandomForestClassifier(
                criterion='entropy',
                n_estimators=n_estimators,
                max_features=max_features,
                min_samples_split=min_samples_split,
                oob_score=True, 
                random_state=42,
                n_jobs=-1 #cpu cores 
            )
            
            # Train the model
            rf.fit(X_train, y_train)
            
            # Predict on all sets
            train_pred = rf.predict(X_train)
            test_pred = rf.predict(X_test)
            val_pred = rf.predict(X_val)
            
            # Calculate accuracies
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            val_acc = accuracy_score(y_val, val_pred)
            oob_acc = rf.oob_score_  # Out-of-bag accuracy
            
            
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  OOB Accuracy:   {oob_acc:.4f}")
            print(f"  Val Accuracy:   {val_acc:.4f}")
            print(f"  Test Accuracy:  {test_acc:.4f}")
            
            # Track best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {
                    'n_estimators': n_estimators,
                    'max_features': max_features,
                    'min_samples_split': min_samples_split
                }
                best_result={
                    'train_accuracy':train_acc,
                    'oob_accuracy':oob_acc,
                    'val_accuracy':val_acc,
                    'test_accuracy':test_acc
                }
                best_model = rf
                best_test_predictions = test_pred
            
            print()

# SAVE RESULTS 

print("GRID SEARCH COMPLETE!")
print()

output_dir = r"C:\IITD\Sem-5\COL774\Assignment 3\outputs"
os.makedirs(output_dir, exist_ok=True)


# ==== BEST MODEL RESULTS ===

print("BEST MODEL (Based on Validation Accuracy)")
print(f"n_estimators:      {best_params['n_estimators']}")
print(f"max_features:      {best_params['max_features']}")
print(f"min_samples_split: {best_params['min_samples_split']}")
print()

# Get accuracies for best model
print(f"Training Accuracy:     {best_result['train_accuracy']:.4f}")
print(f"Out-of-Bag Accuracy:   {best_result['oob_accuracy']:.4f}")
print(f"Validation Accuracy:   {best_result['val_accuracy']:.4f}")
print(f"Test Accuracy:         {best_result['test_accuracy']:.4f}")
print()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, best_test_predictions)
print("Confusion Matrix (Test Set):")
print(conf_matrix)
print()

print("Classification Report (Test Set):")
print(classification_report(y_test, best_test_predictions))
print()

# Save best model predictions
predictions_df = pd.DataFrame(best_test_predictions, columns=['result'])
predictions_df.to_csv(os.path.join(output_dir, 'predictions_f.csv'), header=True, index=False)
print(f"Best model predictions saved to: predictions_f.csv")
print()