
# c.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
file_path = r'C:\IITD\Sem-5\COL774\Assignment 3\decision_tree.py'

# Load the module
spec = importlib.util.spec_from_file_location("decision_tree", file_path)
decision_tree = importlib.util.module_from_spec(spec)
sys.modules["decision_tree"] = decision_tree
spec.loader.exec_module(decision_tree)
# Use the class
DecisionTree= decision_tree.DecisionTree
from decision_tree import DecisionTree
train_df=pd.read_csv(r'C:\IITD\Sem-5\COL774\Assignment 3\train.csv')
test_df=pd.read_csv(r'C:\IITD\Sem-5\COL774\Assignment 3\test.csv')
val_df=pd.read_csv(r'C:\IITD\Sem-5\COL774\Assignment 3\validation.csv')
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
X_train = train_df.drop('result', axis=1)
y_train = train_df['result']

X_test = test_df.drop('result', axis=1)
y_test = test_df['result']

X_val = val_df.drop('result', axis=1)
y_val = val_df['result']
depths=[15,25,35,45]
ccp_alphas=[0.0,0.0001,0.0003,0.0005]
train_accuracies = []
test_accuracies = []
val_accuracies=[]
all_predictions=[]
best_val_acc=0
best_val_predictions=None
# iterate throught depths and get the accuracy for each model
for depth in depths:
    print(f"Training with max_depth={depth}...")
    
    # Create decision tree with entropy criterion
    dt = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=depth,
        random_state=42
    )
    
    # Train the model
    dt.fit(X_train, y_train)
    
    # Predict on all sets
    train_pred = dt.predict(X_train)
    test_pred = dt.predict(X_test)
    val_pred = dt.predict(X_val)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    val_accuracies.append(val_acc)
    # update only if val acc is best unitl now
    if val_acc>best_val_acc:
        best_test_predictions=test_pred

# concat the predictions
all_predictions+=list(best_test_predictions)

# Plot Accuracy
plt.figure(figsize=(12, 8))
plt.plot(depths, train_accuracies, marker='o', label='Training Accuracy', 
            linewidth=2, markersize=8)
plt.plot(depths, test_accuracies, marker='s', label='Testing Accuracy', 
            linewidth=2, markersize=8)
plt.plot(depths, val_accuracies, marker='^', label='Validation Accuracy', 
            linewidth=2, markersize=8)
plt.xlabel('Depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy vs depth', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_vs_nodes_e_depth.png', dpi=300, bbox_inches='tight')
plt.show()

train_accuracies = []
test_accuracies = []
val_accuracies=[]
best_val_acc=0
best_val_predictions=None
# iterate throught ccp_alphas and get the accuracy for each model
for ccp_alpha in ccp_alphas:
    print(f"Training with ccp_alpha={ccp_alpha}...")
    
    # Create decision tree with entropy criterion
    dt = DecisionTreeClassifier(
        criterion='entropy',
        ccp_alpha=ccp_alpha,
        random_state=42
    )
    
    # Train the model
    dt.fit(X_train, y_train)
    
    # Predict on all sets
    train_pred = dt.predict(X_train)
    test_pred = dt.predict(X_test)
    val_pred = dt.predict(X_val)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    val_accuracies.append(val_acc)
    if val_acc>best_val_acc:
        best_test_predictions=test_pred

# concat the predictions
all_predictions+=list(best_test_predictions)
# Plot Accuracy
plt.figure(figsize=(12, 8))
plt.plot(ccp_alphas, train_accuracies, marker='o', label='Training Accuracy', 
            linewidth=2, markersize=8)
plt.plot(ccp_alphas, test_accuracies, marker='s', label='Testing Accuracy', 
            linewidth=2, markersize=8)
plt.plot(ccp_alphas, val_accuracies, marker='^', label='Validation Accuracy', 
            linewidth=2, markersize=8)
plt.xlabel('CCP_Alpha', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy vs ccp_alpha', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_vs_nodes_e_alpha.png', dpi=300, bbox_inches='tight')
plt.show()

# save the predictions into a csv file
predictions_df = pd.DataFrame(all_predictions, columns=['result'])
output_dir = r"C:\IITD\Sem-5\COL774\Assignment 3\outputs"
os.makedirs(output_dir, exist_ok=True)
predictions_df.to_csv(os.path.join(output_dir, 'predictions_e.csv'), header=True, index=False)


