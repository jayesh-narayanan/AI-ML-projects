# b.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util
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
train_df=pd.read_csv(r'C:\IITD\Sem-5\COL774\Assignment 3\train.csv')
test_df=pd.read_csv(r'C:\IITD\Sem-5\COL774\Assignment 3\test.csv')
val_df=pd.read_csv(r'C:\IITD\Sem-5\COL774\Assignment 3\validation.csv')
temp_tree = DecisionTree()
temp_tree.feature_types(train_df)
categorical_cols = [col for col, vals in temp_tree.categorical_info.items() if len(vals) > 2]

# Perform one-hot encoding first before training
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

depths=[15,25,35,45,55]
train_accuracies = []
test_accuracies = []
all_predictions=[]
best_val_acc=0
best_val_predictions=None
# iterate throught depths and get the accuracy for each model
for depth in depths:
    tree=DecisionTree()
    tree.feature_types(train_df)
    tree.fit(train_df,0,depth)
    train_acc=tree.accuracy(train_df)
    test_acc=tree.accuracy(test_df)
    val_acc=tree.accuracy(val_df)
    print(f'training accuracy for depth {depth}: {train_acc}')
    print(f'testing accuracy for depth {depth}: {test_acc}')
    print(f'validation accuracy for depth {depth}: {val_acc}')
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    # update if val_acc is better than the best until now
    if val_acc>best_val_acc:
        best_test_predictions=tree.predict(test_df)


# Plot Accuracy
plt.figure(figsize=(12, 8))
plt.plot(depths, train_accuracies, marker='o', label='Training Accuracy', 
            linewidth=2, markersize=8)
plt.plot(depths, test_accuracies, marker='s', label='Testing Accuracy', 
            linewidth=2, markersize=8)
plt.xlabel('Tree depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy vs Depth', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_vs_depth_b.png', dpi=300, bbox_inches='tight')
plt.show()

# save the predictions into a csv file
predictions_df = pd.DataFrame(best_test_predictions, columns=['result'])
output_dir = r"C:\IITD\Sem-5\COL774\Assignment 3\outputs"
os.makedirs(output_dir, exist_ok=True)
predictions_df.to_csv(os.path.join(output_dir, 'predictions_b.csv'), header=True, index=False)


