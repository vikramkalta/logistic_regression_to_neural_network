import numpy as np
from visualization import create_squiggly_data, plot_dataset, plot_decision_boundary

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        """
        # Add bias term to X
        n_samples, n_features = X.shape
        X = np.hstack((np.ones((n_samples, 1)), X))
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features + 1)
        
        # Gradient descent
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights)
            predictions = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            
            # Update weights
            self.weights -= self.learning_rate * dw
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        # Add bias term to X
        n_samples = X.shape[0]
        X = np.hstack((np.ones((n_samples, 1)), X))
        
        linear_model = np.dot(X, self.weights)
        y_pred = self.sigmoid(linear_model)
        
        # Convert probabilities to binary predictions
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_class)

def main():
    # Generate dataset
    X, y = create_squiggly_data()
    
    # Visualize the dataset
    print("\nOriginal Dataset:")
    plot_dataset(X, y)
    
    # Train logistic regression
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    # Visualize the decision boundary
    print("\nDecision Boundary:")
    plot_decision_boundary(X, y, model)
    
    # Make predictions
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nModel Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
