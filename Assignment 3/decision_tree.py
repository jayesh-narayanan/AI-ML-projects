import numpy as np
import pandas as pd
import math

class TreeNode:
    def __init__(self):
        self.feature = None  # feature about which to split
        self.value = None    # 0.5 in case of boolean or numerical data
        self.children = []
        self.isleaf = False
        self.depth = 0
        self.ans = -1        # the answer in case of leafnode

class DecisionTree:
    def __init__(self, criterion="entropy", min_samples_split=2, min_samples_leaf=1):
        """Initializes the DecisionTree classifier."""
        self.root = None
        self.criterion = criterion
        self.categorical_feature = {}
        self.categorical_info = {}
        self.boolean_feature = {}
        self.min_samples_split = min_samples_split  #  minimum samples to split
        self.min_samples_leaf = min_samples_leaf    #  minimum samples in leaf

    def fit(self, df, depth=0, max_depth=None):
        """Construct the decision tree"""
        # initialize the root and get data on features
        root = TreeNode()
        if depth == 0:
            self.root = root
            self.feature_types(df)  
        
        # Stop if there are no more samples or max_depth exceeded
        if len(df) < self.min_samples_split or (max_depth is not None and depth >= max_depth):
            root.isleaf = True
            root.ans = self.majority(df)
            root.depth = depth
            return root
        
        # pure node checking
        if len(df['result'].unique()) == 1:
            root.isleaf = True
            root.ans = df['result'].iloc[0]
            root.depth = depth
            return root
        
        opt_ans = -1
        opt_gain = 0.0
        opt_values = None
        opt_children = None
        
        entropy_before_split = self.impurity(df)
        
        for col in df.columns:
            # exclude the target col
            if col != 'result':
                if self.categorical_feature[col]:
                    mi = 0.0
                    categorical_children = []
                    major_class = None
                    major_size = -1
                    has_empty_subset = False  
                    # getting entropy for this split
                    for value, subset in df.groupby(col):
                        weight = len(subset) / len(df)
                        mi += weight * self.impurity(subset)
                        categorical_children.append(subset)
                        
                        if len(subset) > major_size:
                            major_size = len(subset)
                            major_class = self.majority(subset)
                        
                        # Check if any child would be too small
                        if len(subset) < self.min_samples_leaf:
                            has_empty_subset = True
                    # get gain for this split
                    gain = entropy_before_split - mi
                    
                    # Only consider split if NO empty subsets and gain is positive 
                    # update if gain is best unitl now
                    if not has_empty_subset and gain > opt_gain:
                        opt_ans = major_class
                        opt_gain = gain
                        opt_values = (col, self.categorical_info[col])
                        opt_children = categorical_children

                elif self.boolean_feature[col]:
                    mid = 0.5
                    split1, split2 = self.splitdata_2(df, col, mid)
                    
                    # Check minimum samples
                    if len(split1) < self.min_samples_leaf or len(split2) < self.min_samples_leaf:
                        continue
                    # get gain for this split
                    p = float(len(split1)) / len(df)
                    mi = p * self.impurity(split1) + (1 - p) * self.impurity(split2)
                    gain = entropy_before_split - mi
                    # update gain if best until now
                    if gain > opt_gain:
                        opt_ans = self.majority(df)
                        opt_gain = gain
                        opt_values = (col, mid)
                        opt_children = [split1, split2]
                else:
                    arr = df[col]
                    arr = np.array(arr)
                    mid = np.median(arr)
                    # split about median
                    split1, split2 = self.splitdata_2(df, col, mid)
                    
                    # Check minimum samples
                    if len(split1) < self.min_samples_leaf or len(split2) < self.min_samples_leaf:
                        continue
                    
                    p = float(len(split1)) / len(df)
                    mi = p * self.impurity(split1) + (1 - p) * self.impurity(split2)
                    gain = entropy_before_split - mi
                    # update 
                    if gain > opt_gain:
                        opt_ans = self.majority(df)
                        opt_gain = gain
                        opt_values = (col, mid)
                        opt_children = [split1, split2]
        
        # Only split if gain is above threshold
        if opt_gain > 1e-7:  # Small epsilon to avoid numerical issues
            root.ans = opt_ans
            root.depth = depth
            root.feature = opt_values[0]
            root.value = opt_values[1]
            for subset in opt_children:
                # do rec and add the node to the .children of the root node
                root.children.append(self.fit(subset, depth + 1, max_depth))
            return root
        else:
            # if invalid or leaf is reached
            root.isleaf = True
            root.ans = self.majority(df)
            root.depth = depth
            return root

    def feature_types(self, df):
        for col in df.columns:
            if col == 'result':
                continue
                
            dtype = df[col].dtype
            # check if feature is categorical
            if pd.api.types.is_string_dtype(dtype):
                self.categorical_feature[col] = True
                unique_vals = df[col].dropna().unique().tolist()
                self.categorical_info[col] = unique_vals
            else:
                self.categorical_feature[col] = False
            # check if feature is boolean
            if pd.api.types.is_bool_dtype(dtype) or (set(df[col].unique()) <= {0, 1}):
                self.boolean_feature[col] = True
            else:
                self.boolean_feature[col] = False

    def entropy(self, data):
        y = (data['result'] == 1).sum()
        n = (data['result'] != 1).sum()
        total = y + n

        if total == 0:
            return 0
        p_y = y / total
        p_n = n / total
        entropy = 0
        if p_y > 0:
            entropy -= p_y * math.log2(p_y)
        if p_n > 0:
            entropy -= p_n * math.log2(p_n)

        return entropy
    
    def gini(self, data):
        y = (data['result'] == 1).sum()
        n = (data['result'] != 1).sum()
        total = y + n

        if total == 0:
            return 0
        p_y = y / total
        p_n = n / total
        gini = 1 - (p_y**2 + p_n**2)
        return gini
    
    def impurity(self, data):
        if self.criterion == 'entropy':
            return self.entropy(data)
        elif self.criterion == 'gini':
            return self.gini(data)
    
    def majority(self, data):
        y = (data['result'] == 1).sum()
        n = (data['result'] != 1).sum()
        return 1 if y > n else 0
    
    def splitdata_2(self, df, col, mid):
        split1 = df[df[col] <= mid].copy()
        split2 = df[df[col] > mid].copy()
        return split1, split2

    def post_prune(self, validation_df):
        """
        Post-pruning with proper backup and restore
        """
        current_acc = self.accuracy(validation_df)
        improved = True
        iteration = 0

        while improved:
            improved = False
            best_gain = 0
            best_node = None
            # iterate among available nodes O(n^2)
            nodes = self.get_prunable_nodes()
            for node in nodes:
                # Backup both isleaf AND children
                backup_isleaf = node.isleaf
                backup_children = node.children
                backup_feature = node.feature
                backup_value = node.value

                # Temporarily prune
                node.isleaf = True
                node.children = []

                new_acc = self.accuracy(validation_df)
                gain = new_acc - current_acc

                # restore pruned node completely
                node.isleaf = backup_isleaf
                node.children = backup_children
                node.feature = backup_feature
                node.value = backup_value

                if gain > best_gain:
                    best_gain = gain
                    best_node = node

            # If best_node improves accuracy, prune the node that was temporarily pruned, permanently
            if best_node and best_gain > 0:
                iteration += 1
                print(f"[Iteration {iteration}] Pruned node at depth {best_node.depth} "
                      f"with accuracy gain {best_gain:.4f}")
                best_node.isleaf = True
                best_node.children = []
                current_acc += best_gain
                improved = True

        print(f"Final validation accuracy after pruning: {current_acc:.4f}")
        return current_acc

    def get_prunable_nodes(self, node=None):
        """Get all non-leaf nodes that can be pruned"""
        if node is None:
            node = self.root

        nodes = []
        if not node.isleaf and node.children:
            all_children_leaves = all(child.isleaf for child in node.children)
            if all_children_leaves:
                nodes.append(node)
            else:
                for child in node.children:
                    # do recursion
                    nodes.extend(self.get_prunable_nodes(child))
        return nodes

    
    def number_of_nodes(self,node):
        if node.isleaf:
            return 1
        total=1
        for i in range(len(node.children)):
            total+=self.number_of_nodes(node.children[i])
        return total

    def predict(self, df):
        """
        Predict class labels for a dataframe
        """
        # Convert DataFrame columns to numpy arrays 
        data_dict = {col: df[col].values for col in df.columns if col != 'result'}
        n_samples = len(df)
        
        predictions = np.zeros(n_samples, dtype=int)
        
        # Predict each sample
        for i in range(n_samples):
            sample = {col: data_dict[col][i] for col in data_dict}
            predictions[i] = self.predict_row(sample)
        
        return predictions


    def predict_row(self, entry):
        """
        Predict single row using while loop 
        """
        current = self.root
        
        # Use while loop instead of recursion 
        while not current.isleaf:
            feature = current.feature
            
            if self.categorical_feature[feature]:
                entry_value = entry[feature]
                

                # search through children
                found = False
                for i in range(min(len(current.children), len(current.value))):
                    if current.value[i] == entry_value:
                        current = current.children[i]
                        found = True
                        break
                
                if not found:
                    return current.ans
                
            else:
                # Numerical feature
                if entry[feature] <= current.value:
                    current = current.children[0]
                else:
                    current = current.children[1]
        
        return current.ans


    def accuracy(self, df):
        """
        Calculate accuracy by predicting and comparing
        """
        predictions = self.predict(df)
        y_true = df['result'].values
        return np.mean(predictions == y_true)
