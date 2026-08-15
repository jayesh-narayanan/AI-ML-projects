from neural_network import NeuralNetwork
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import pickle
# ========== Helper Functions ==========

def load_all_images_from_folder(base_dir, img_size=(32, 32)):
    """
    Load all images from folder structure where each subfolder is a class.
    Converts images to grayscale and normalizes to [0, 1].
    
    Args:
        base_dir: Base directory containing class subfolders
        img_size: Tuple of (height, width) for resizing images
    
    Returns:
        X: numpy array of shape (n_samples, n_features) - grayscale flattened
        y: numpy array of shape (n_samples,) with class indices
        num_classes: number of classes
    """
    all_images = []
    all_labels = []
    class_names = sorted(os.listdir(base_dir))
    
    print(f"Loading images from {base_dir}...")
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Iterate through each class
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(base_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue
        
        # only load images in number set
        if class_idx<36:
            continue
        class_count = 0
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            try:
                # Open image and convert to RBG
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                img = cv2.resize(img, img_size) 
                img_array = img.astype(np.float32) / 255.0  
                # Flatten to 1D array (32*32*3 = 3072 features for grayscale)
                all_images.append(img_array.flatten())
                all_labels.append(class_idx)
                class_count += 1
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        print(f"  Class {class_idx} ({class_name}): {class_count} images")
    
    # Convert to numpy arrays
    X = np.array(all_images)
    y = np.array(all_labels).reshape(-1, 1)
    
    # Get number of classes
    num_classes = len([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    print(f"Total images loaded: {len(X)}")
    print(f"Image shape after flattening: {X.shape}")
    print(f"Features per image: {X.shape[1]} (32x32 grayscale)")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {num_classes}")
    print()
    
    return X, y, num_classes


def evaluate_model(nn, X, y_encoded):
    """
    Evaluate model and print metrics.
    
    Args:
        nn: Neural network model
        X: Input data
        y_encoded: One-hot encoded labels
    
    Returns:
        accuracy, precision, recall, f1
    """
    predictions = nn.predict(X)
    y_true = np.argmax(y_encoded, axis=1)
    
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average='macro', zero_division=0)
    recall = recall_score(y_true, predictions, average='macro', zero_division=0)
    f1 = f1_score(y_true, predictions, average='macro', zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    return accuracy, precision, recall, f1



# ========== Activation Functions ==========

def relu(z):
    return np.maximum(z, 0.0)

def delRelu(z):
    return 1.*(z > 0)

def sigmoid(z):
    return 1/(1 + np.exp(-np.clip(z, -500, 500)))  # Added clipping for stability

def delSigmoid(z):
    return z*(1 - z)

# ========== Main Training Function ==========

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load the data from each folder corresponding to a class
    train_dir = r"C:\IITD\Sem-5\COL774\Assignment 3\neural_network\train"
    test_dir = r"C:\IITD\Sem-5\COL774\Assignment 3\neural_network\test"
    
    # Load images
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    X_train, y_train, num_classes = load_all_images_from_folder(train_dir)
    X_test, y_test, _ = load_all_images_from_folder(test_dir)
    
    # One-hot encode labels using sklearn
    print("One-hot encoding labels...")
    enc = OneHotEncoder(sparse_output=False)
    y_train_encoded = enc.fit_transform(y_train)
    y_test_encoded = enc.transform(y_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train_encoded.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test_encoded.shape}")
    print()
    
    # Different hidden layer architectures to experiment with
    hidden_arch=[512,256,128,64]
    
    # List to store ALL predictions 
    all_predictions = []

    # Training from scratch
    print(f"Training from scratch")
    
    # Initialize neural network
    nn = NeuralNetwork(
        xTrain=X_train,
        yTrain=y_train_encoded,
        layers=hidden_arch,         
        activation=relu,         
        delActivation=delRelu,    
        learningRate=0.01,
        batchSize=32,
        maxIterations=20,        
        epi=1e-2,
        only_epochs=True # train for exactly 20 epochs irrespective of tolerance
    )
    
    # Train the neural network
    print("\nTraining...")
    start_time = time.time()
    iterations, final_loss, f1_train_scores, f1_test_scores= nn.train(xTest=X_test,yTest=y_test_encoded)
    end_time = time.time()
    
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    print(f"Iterations: {iterations}, Final Loss: {final_loss:.6f}")
    
    # Evaluate on training and test sets
    print()
    print(f"RESULTS FOR Training from scratch:")
    print("Training metrics:")
    train_acc, train_prec, train_rec, train_f1 = evaluate_model(nn, X_train, y_train_encoded)
    
    print()
    print("Testing metrics:")
    test_acc, test_prec, test_rec, test_f1 = evaluate_model(nn, X_test, y_test_encoded)

    print()
    print("Training Confusion Matrix:")
    train_conf_matrix = nn.confusionMatrix(X_train, y_train_encoded)
    print(train_conf_matrix)
    
    print()
    print("Testing Confusion Matrix:")
    test_conf_matrix = nn.confusionMatrix(X_test, y_test_encoded)
    print(test_conf_matrix)
    print()

    # Get predictions on test set
    test_predictions = nn.predict(X_test)
    all_predictions.extend(test_predictions.tolist())

    # Plot F1 Score
    epochs = range(1, 21)
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, f1_train_scores, marker='o', label='Training F1', 
             linewidth=2, markersize=8)
    plt.plot(epochs, f1_test_scores, marker='s', label='Testing F1', 
             linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Avg F1 score', fontsize=12)
    plt.title('Avg F1 score vs epochs (scratch)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('f1_scores_vs_epochs_scratch.png', dpi=300, bbox_inches='tight')
    plt.show()

    #Training using fine-tuning
    # load hidden layer weights from the pickle file
    print("Training by fine-tuning the neural-network from part d")
    filepath=r"C:\IITD\Sem-5\COL774\Assignment 3\neural_network\weights.pkl"
    with open(filepath, 'rb') as f:
        weights_dict = pickle.load(f)
    
    saved_weights = weights_dict['weights']
    saved_biases = weights_dict['biases']

    
    # Initialize new network
    nn = NeuralNetwork(
        xTrain=X_train,
        yTrain=y_train_encoded,
        layers=hidden_arch,          # Hidden layer architecture
        activation=relu,          # Can also try relu
        delActivation=delRelu,    # Can also try delRelu
        learningRate=0.01,
        batchSize=32,
        maxIterations=20,           # Max Number of epochs
        epi=1e-2,
        only_epochs=True
    )
    # copy the weights and biases into this model
    for i in range(len(saved_weights) - 1):
        nn.weights[i] = saved_weights[i].copy()
        nn.biases[i] = saved_biases[i].copy()
    
    # zero-initialize output layer
    output_idx = len(nn.weights) - 1
    nn.weights[output_idx] = np.zeros_like(nn.weights[output_idx])
    nn.biases[output_idx] = np.zeros_like(nn.biases[output_idx])

    # Train the neural network
    print("\nTraining...")
    start_time = time.time()
    iterations, final_loss, f1_train_scores, f1_test_scores= nn.train(xTest=X_test,yTest=y_test_encoded)
    end_time = time.time()
    
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    print(f"Iterations: {iterations}, Final Loss: {final_loss:.6f}")
    
    # Evaluate on training and test sets
    print()
    print(f"RESULTS FOR Training from scratch:")
    print("Training metrics:")
    train_acc, train_prec, train_rec, train_f1 = evaluate_model(nn, X_train, y_train_encoded)
    
    print()
    print("Testing metrics:")
    test_acc, test_prec, test_rec, test_f1 = evaluate_model(nn, X_test, y_test_encoded)

    print()
    print("Training Confusion Matrix:")
    train_conf_matrix = nn.confusionMatrix(X_train, y_train_encoded)
    print(train_conf_matrix)
    
    print()
    print("Testing Confusion Matrix:")
    test_conf_matrix = nn.confusionMatrix(X_test, y_test_encoded)
    print(test_conf_matrix)
    print()

    # Get predictions on test set
    test_predictions = nn.predict(X_test)
    all_predictions.extend(test_predictions.tolist())
    print()

    # Plot F1 Score
    epochs = range(1, 21)
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, f1_train_scores, marker='o', label='Training F1', 
             linewidth=2, markersize=8)
    plt.plot(epochs, f1_test_scores, marker='s', label='Testing F1', 
             linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Avg F1 score', fontsize=12)
    plt.title('Avg F1 score vs epochs (fine-tuned)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('f1_scores_vs_epochs_fine-tuned.png', dpi=300, bbox_inches='tight')
    plt.show()



    # save the predictions into a csv file
    predictions_df = pd.DataFrame(all_predictions, columns=['prediction'])
    output_dir = r"C:\IITD\Sem-5\COL774\Assignment 3\neural_network\outputs"
    os.makedirs(output_dir, exist_ok=True)
    predictions_df.to_csv(os.path.join(output_dir, 'predictions_f.csv'), header=True, index=False)
    
    
    

    

