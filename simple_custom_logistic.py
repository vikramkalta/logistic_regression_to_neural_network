"""
Simple Custom Logistic Regression - Mathematical Implementation
============================================================

This demonstrates the core mathematical concepts of logistic regression:
1. Sigmoid function implementation
2. Cost function calculation
3. Gradient descent step by step
4. Manual prediction process
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

class SimpleLogisticRegression:
    """
    Simplified logistic regression implementation showing the math clearly.
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=100):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """
        Sigmoid function: σ(z) = 1 / (1 + e^(-z))
        """
        # Prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, y_true, y_pred):
        """
        Logistic regression cost function:
        J = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
        """
        m = len(y_true)
        # Prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -(1/m) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return cost
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        """
        # Initialize parameters
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        print("🧮 Mathematical Training Process:")
        print("=" * 50)
        print(f"Initial weights: {self.weights}")
        print(f"Initial bias: {self.bias}")
        print(f"Learning rate: {self.learning_rate}")
        print()
        
        # Training loop
        for i in range(self.max_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Compute cost
            cost = self.compute_cost(y, predictions)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1/m) * np.dot(X.T, (predictions - y))
            db = (1/m) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print every 20 iterations
            if i % 20 == 0:
                print(f"Iteration {i:2d}: Cost = {cost:.4f}, Weights = {self.weights.round(3)}, Bias = {self.bias:.3f}")
        
        print(f"\n✅ Final parameters:")
        print(f"   Weights: {self.weights.round(4)}")
        print(f"   Bias: {self.bias:.4f}")
    
    def predict_step_by_step(self, X):
        """
        Show the prediction process step by step.
        """
        print("\n🔍 Step-by-Step Prediction Process:")
        print("=" * 50)
        
        for i in range(min(3, len(X))):  # Show first 3 samples
            x = X[i]
            print(f"\nSample {i+1}: {x.round(3)}")
            
            # Step 1: Linear combination
            z = np.dot(x, self.weights) + self.bias
            print(f"  Step 1 - Linear combination:")
            print(f"    z = w₁×x₁ + w₂×x₂ + b")
            print(f"    z = {self.weights[0]:.3f}×{x[0]:.3f} + {self.weights[1]:.3f}×{x[1]:.3f} + {self.bias:.3f}")
            print(f"    z = {z:.3f}")
            
            # Step 2: Sigmoid transformation
            prob = self.sigmoid(z)
            print(f"  Step 2 - Sigmoid transformation:")
            print(f"    σ(z) = 1 / (1 + e^(-z))")
            print(f"    σ({z:.3f}) = 1 / (1 + e^(-{z:.3f}))")
            print(f"    σ({z:.3f}) = {prob:.3f}")
            
            # Step 3: Classification
            prediction = 1 if prob >= 0.5 else 0
            print(f"  Step 3 - Classification:")
            print(f"    Prediction = {prediction} (probability = {prob:.3f})")
            print(f"    Class: {'Positive' if prediction == 1 else 'Negative'}")


def demonstrate_mathematical_concepts():
    """
    Demonstrate the mathematical concepts with visualizations.
    """
    print("🌟 CUSTOM LOGISTIC REGRESSION MATHEMATICS")
    print("=" * 60)
    
    # Create a simple 2D dataset
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                              n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    
    # 1. Visualize the sigmoid function
    print("\n📊 1. Sigmoid Function Visualization")
    z_range = np.linspace(-10, 10, 100)
    sigmoid_values = 1 / (1 + np.exp(-z_range))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(z_range, sigmoid_values, 'b-', linewidth=3)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.title('Sigmoid Function\nσ(z) = 1/(1+e^(-z))')
    plt.grid(True, alpha=0.3)
    
    # 2. Show the dataset
    plt.subplot(1, 3, 2)
    plt.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], c='red', marker='o', alpha=0.7, label='Class 0')
    plt.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], c='blue', marker='s', alpha=0.7, label='Class 1')
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title('Training Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Train the model
    model = SimpleLogisticRegression(learning_rate=0.1, max_iterations=100)
    model.fit(X_scaled, y)
    
    # Plot cost history
    plt.subplot(1, 3, 3)
    plt.plot(model.cost_history, 'g-', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function During Training')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 4. Show step-by-step predictions
    model.predict_step_by_step(X_scaled[:3])
    
    # 5. Visualize decision boundary
    print("\n🎨 Decision Boundary Visualization")
    plt.figure(figsize=(10, 8))
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    z_mesh = np.dot(mesh_points, model.weights) + model.bias
    prob_mesh = model.sigmoid(z_mesh)
    prob_mesh = prob_mesh.reshape(xx.shape)
    
    # Plot decision boundary and probabilities
    plt.contourf(xx, yy, prob_mesh, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Probability')
    plt.contour(xx, yy, prob_mesh, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    
    # Plot data points
    plt.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], c='red', marker='o', 
               edgecolors='black', s=50, alpha=0.8, label='Class 0')
    plt.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], c='blue', marker='s', 
               edgecolors='black', s=50, alpha=0.8, label='Class 1')
    
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title('Decision Boundary and Probability Map')
    plt.legend()
    plt.show()
    
    # 6. Mathematical interpretation
    print("\n🔍 Mathematical Interpretation:")
    print("=" * 40)
    print(f"Decision boundary equation:")
    print(f"{model.weights[0]:.3f} × x₁ + {model.weights[1]:.3f} × x₂ + {model.bias:.3f} = 0")
    print(f"\nFor classification:")
    print(f"- If z > 0 → σ(z) > 0.5 → Predict Class 1")
    print(f"- If z < 0 → σ(z) < 0.5 → Predict Class 0")
    print(f"- If z = 0 → σ(z) = 0.5 → Decision boundary")
    
    return model


if __name__ == "__main__":
    model = demonstrate_mathematical_concepts()
    
    print("\n🎉 Custom Mathematical Implementation Complete!")
    print("=" * 60)
    print("You've seen:")
    print("✅ Sigmoid function implementation")
    print("✅ Cost function calculation")
    print("✅ Gradient descent step-by-step")
    print("✅ Manual prediction process")
    print("✅ Decision boundary visualization")
    print("✅ Mathematical interpretation")
