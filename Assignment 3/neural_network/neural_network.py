import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

# ==== Activation Functions =====

def relu(z):
    return np.maximum(z, 0.0)

def delRelu(z):
    return 1.*(z > 0)

def sigmoid(z):
    return 1/(1 + np.exp(-np.clip(z, -500, 500)))

def delSigmoid(z):
    return z*(1 - z)

def softmax(z):
    """Softmax activation"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(Y_true, Y_pred):
    """Cross-entropy loss"""
    Y_pred_clipped = np.clip(Y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(Y_true * np.log(Y_pred_clipped)) / Y_true.shape[0]

class NeuralNetwork:
    """
    Neural Network Implementation
    """
    
    def __init__(self, xTrain, yTrain, layers, activation, delActivation, 
                 learningRate=0.01, epi=1e-3, batchSize=100, maxIterations=1000, 
                 verbose=False, mse=False, only_epochs=False):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.layers = [xTrain.shape[1]] + layers + [yTrain.shape[1]]
        self.activation = activation
        self.delActivation = delActivation
        self.learningRate = learningRate
        self.epi = epi
        self.maxIterations = maxIterations
        self.batchSize = batchSize
        self.only_epochs = only_epochs
        self.weights = []
        self.biases = []
        self.mse = mse
        
        # initialize the weights and bias with randm values
        for i in range(1, len(self.layers)):
            self.weights.append(
                np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2.0 / self.layers[i-1])
            )
            self.biases.append(
                np.random.randn(self.layers[i], 1) * np.sqrt(2.0 / self.layers[i-1])
            )
        
        self.num_layers = len(self.layers) - 1
        self.delta = [None] * self.num_layers

    def train(self, xTest=None, yTest=None):
        """
        training method with other methods called inside it
        """
        m = self.xTrain.shape[0]
        
        # permute the data while training for diversity
        p = np.random.permutation(m)
        xTrain = self.xTrain[p]
        yTrain = self.yTrain[p]
        
        prevErr = 1e9
        itr = 0
        
        # Determine whether part f or not
        track_f1 = self.maxIterations <= 50
        f1_train_scores = [] if track_f1 else None
        f1_test_scores = [] if track_f1 else None
        
        M = self.batchSize
        b = m // M
        rate = self.learningRate
        
        
        # maxitrs==maxepochs
        while itr < self.maxIterations:
            avgErr = 0
            
            for i in range(b):
        
                X = xTrain[i*M:(i+1)*M]
                Y = yTrain[i*M:(i+1)*M]

                allOutputs = [X]
                # forward propogation to get the output of each layer
                allOutputs=self.forward(allOutputs)
                
                # back propogation to return the delta of loss in each layer
                self.backward(allOutputs,Y)
                
                # update parameters
                self.update_params(allOutputs,M)
                
                # accumulate loss for all output neurons
                avgErr += cross_entropy_loss(Y, allOutputs[-1])
            
            avgErr *= (1.0 / b)  
            itr += 1
            
            # PRINT PROGRESS 
            if track_f1:
                # Only compute predictions when needed
                trainAcc = self.findAcc(self.xTrain, self.yTrain)
                
                y_true_train = np.argmax(self.yTrain, axis=1)
                y_pred_train = self.predict(self.xTrain)
                f1_train = f1_score(y_true_train, y_pred_train, average='macro', zero_division=0)
                f1_train_scores.append(f1_train)
                
                if xTest is not None and yTest is not None:
                    y_true_test = np.argmax(yTest, axis=1)
                    y_pred_test = self.predict(xTest)
                    f1_test = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)
                    f1_test_scores.append(f1_test)
                    
                    print(f"Epoch {itr}/{self.maxIterations}, Loss: {avgErr:.6f}, "
                          f"Acc: {trainAcc:.4f}, F1_train: {f1_train:.4f}, F1_test: {f1_test:.4f}")
                else:
                    print(f"Epoch {itr}/{self.maxIterations}, Loss: {avgErr:.6f}, "
                          f"Acc: {trainAcc:.4f}, F1_train: {f1_train:.4f}")
            else:
                if itr % 10 == 0:
                    trainAcc = self.findAcc(self.xTrain, self.yTrain)
                    print(f"Iteration {itr}/{self.maxIterations}, Loss: {avgErr:.6f}, Train Acc: {trainAcc:.4f}")
            
            # Check for convergence
            if abs(avgErr - prevErr) < self.epi and not self.only_epochs:
                print(f"Converged at iteration {itr}")
                break
            prevErr = avgErr
        
        # Return appropriate values
        if track_f1:
            return itr, avgErr, f1_train_scores, f1_test_scores
        else:
            return itr, avgErr
    
    def forward(self,allOutputs):
        # store output of jth layer in allOutputs[j] and do the process iteratively till last layer
        for j in range(self.num_layers):
            net = allOutputs[j] @ self.weights[j].T + self.biases[j].T
            
            if j < self.num_layers - 1:  # Hidden layers
                allOutputs.append(self.activation(net))
            else:  # Output layer
                allOutputs.append(softmax(net))
        return allOutputs
    
    def backward(self,allOutputs,Y):
        # Output layer delta
        self.delta[self.num_layers - 1] = Y - allOutputs[-1]
        
        # Backpropagate in reverse without list operations to get the derivative of loss in the jth layer
        for j in range(self.num_layers - 2, -1, -1):
            self.delta[j] = (self.delta[j+1] @ self.weights[j+1]) * self.delActivation(allOutputs[j+1])

    def update_params(self,allOutputs,M):
        inv_M = 1.0 / M 
        for j in range(self.num_layers):
            grad = (self.delta[j].T @ allOutputs[j]) * inv_M
            self.weights[j] += self.learningRate* grad
            self.biases[j] += self.learningRate* np.sum(self.delta[j], axis=0, keepdims=True).T * inv_M

    def predict(self, xTest):
        """
        prediction method
        """
        allOutputs = xTest
        
        # iterate though each layer the output of the previous layer is the input to the next layer
        for j in range(self.num_layers):
            net = allOutputs @ self.weights[j].T + self.biases[j].T
            
            if j < self.num_layers - 1:
                allOutputs = self.activation(net)
            else:
                allOutputs = softmax(net)
        
        return np.argmax(allOutputs, axis=1)
    
    def findAcc(self, xTest, yTest):
        return np.mean(self.predict(xTest) == np.argmax(yTest, axis=1))
    
    def confusionMatrix(self, xTest, yTest):
        yPred = self.predict(xTest)
        return confusion_matrix(np.argmax(yTest, axis=1), yPred)
    
    def get_probabilities(self, xTest):
        """
        Get probability distributions (similar to predict but returning the probabs instead of taking mx)
        """
        allOutputs = xTest
        
        for j in range(self.num_layers):
            net = allOutputs @ self.weights[j].T + self.biases[j].T
            
            if j < self.num_layers - 1:
                allOutputs = self.activation(net)
            else:
                allOutputs = softmax(net)
        
        return allOutputs




