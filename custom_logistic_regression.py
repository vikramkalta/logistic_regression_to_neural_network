"""
Custom Logistic Regression Implementation from Scratch
=====================================================

This implementation shows the mathematical foundations of logistic regression:
1. Sigmoid function
2. Cost function (log-likelihood)
3. Gradient descent optimization
4. Step-by-step training visualization
5. Decision boundary visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CustomLogisticRegression:
    """
    Custom implementation of Logistic Regression from scratch.
    
    Mathematical Foundation:
    - Sigmoid: σ(z) = 1 / (1 + e^(-z))
    - Hypothesis: h(x) = σ(θ^T * x)
    - Cost Function: J(θ) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
    - Gradient: ∇J(θ) = (1/m) * X^T * (h(x) - y)
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize the logistic regression model.
        
        Parameters:
        - learning_rate: Step size for gradient descent
        - max_iterations: Maximum number of training iterations
        - tolerance: Convergence tolerance for cost function
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.weight_history = []
        
    def sigmoid(self, z):
        """
        Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
        
        Handles overflow by clipping z values.
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, y_true, y_pred):
        """
        Compute the logistic regression cost function (log-likelihood).
        
        J(θ) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
        """
        m = len(y_true)
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -(1/m) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return cost
    
    def fit(self, X, y, verbose=True):
        """
        Train the logistic regression model using gradient descent.
        
        Parameters:
        - X: Feature matrix (m x n)
        - y: Target vector (m x 1)
        - verbose: Print training progress
        """
        # Initialize parameters
        m, n = X.shape
        self.weights = np.random.normal(0, 0.01, n)
        self.bias = 0
        
        # Store history for visualization
        self.cost_history = []
        self.weight_history = []
        
        print("🚀 Starting Custom Logistic Regression Training...")
        print("=" * 60)
        print(f"Dataset: {m} samples, {n} features")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Max iterations: {self.max_iterations}")
        print()
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass: compute predictions
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Compute cost
            cost = self.compute_cost(y, predictions)
            self.cost_history.append(cost)
            self.weight_history.append(self.weights.copy())
            
            # Compute gradients
            dw = (1/m) * np.dot(X.T, (predictions - y))
            db = (1/m) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if verbose and (i % 100 == 0 or i == self.max_iterations - 1):
                print(f"Iteration {i:4d}: Cost = {cost:.6f}")
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - cost) < self.tolerance:
                print(f"✅ Converged at iteration {i}")
                break
        
        print(f"🎯 Final cost: {self.cost_history[-1]:.6f}")
        print("✅ Training completed!")
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_decision_boundary_line(self, x_range, feature_idx1=0, feature_idx2=1):
        """
        Get decision boundary line for 2D visualization.
        
        For 2D case: w1*x1 + w2*x2 + b = 0
        Solving for x2: x2 = -(w1*x1 + b) / w2
        """
        if len(self.weights) < 2:
            return None
        
        w1, w2 = self.weights[feature_idx1], self.weights[feature_idx2]
        if abs(w2) < 1e-10:  # Avoid division by zero
            return None
        
        x2 = -(w1 * x_range + self.bias) / w2
        return x2


class IrisCustomLogisticRegression:
    """
    Comprehensive tutorial using custom logistic regression on Iris dataset.
    """
    
    def __init__(self):
        """Initialize with Iris dataset."""
        print("🌸 Loading Iris Dataset for Custom Logistic Regression...")
        print("=" * 60)
        
        # Load Iris dataset
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names
        
        # Create DataFrame
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df['species'] = pd.Categorical.from_codes(self.y, self.target_names)
        
        print(f"Dataset loaded: {self.df.shape}")
        print(f"Features: {list(self.feature_names)}")
        print(f"Classes: {list(self.target_names)}")
    
    def visualize_mathematical_functions(self):
        """Visualize the mathematical functions used in logistic regression."""
        print("\n📊 Visualizing Mathematical Functions...")
        print("=" * 50)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Sigmoid function
        z = np.linspace(-10, 10, 100)
        sigmoid_values = 1 / (1 + np.exp(-z))
        
        axes[0, 0].plot(z, sigmoid_values, 'b-', linewidth=3, label='σ(z) = 1/(1+e^(-z))')
        axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Threshold')
        axes[0, 0].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        axes[0, 0].set_xlabel('z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ')
        axes[0, 0].set_ylabel('σ(z)')
        axes[0, 0].set_title('Sigmoid Function')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Cost function behavior
        p = np.linspace(0.001, 0.999, 100)
        cost_y1 = -np.log(p)  # Cost when y=1
        cost_y0 = -np.log(1-p)  # Cost when y=0
        
        axes[0, 1].plot(p, cost_y1, 'r-', linewidth=2, label='Cost when y=1: -log(p)')
        axes[0, 1].plot(p, cost_y0, 'b-', linewidth=2, label='Cost when y=0: -log(1-p)')
        axes[0, 1].set_xlabel('Predicted Probability (p)')
        axes[0, 1].set_ylabel('Cost')
        axes[0, 1].set_title('Logistic Regression Cost Function')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Gradient visualization (conceptual)
        x = np.linspace(-5, 5, 100)
        y = x**2  # Simple quadratic for gradient concept
        gradient = 2*x
        
        axes[1, 0].plot(x, y, 'g-', linewidth=2, label='Cost Function (conceptual)')
        # Show gradient at a few points
        points = [-3, -1, 1, 3]
        for point in points:
            grad_val = 2*point
            axes[1, 0].arrow(point, point**2, -0.5*grad_val, -0.5*grad_val*grad_val, 
                           head_width=0.2, head_length=0.2, fc='red', ec='red')
        
        axes[1, 0].set_xlabel('Parameter Value')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].set_title('Gradient Descent Concept')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Learning rate effect
        iterations = np.arange(0, 50)
        lr_high = 0.1
        lr_medium = 0.01
        lr_low = 0.001
        
        # Simulated convergence curves
        cost_high = 10 * np.exp(-lr_high * iterations) + np.random.normal(0, 0.1, len(iterations))
        cost_medium = 10 * np.exp(-lr_medium * iterations) + np.random.normal(0, 0.05, len(iterations))
        cost_low = 10 * np.exp(-lr_low * iterations) + np.random.normal(0, 0.02, len(iterations))
        
        axes[1, 1].plot(iterations, cost_high, 'r-', label=f'High LR ({lr_high})', alpha=0.8)
        axes[1, 1].plot(iterations, cost_medium, 'g-', label=f'Medium LR ({lr_medium})', alpha=0.8)
        axes[1, 1].plot(iterations, cost_low, 'b-', label=f'Low LR ({lr_low})', alpha=0.8)
        axes[1, 1].set_xlabel('Iterations')
        axes[1, 1].set_ylabel('Cost')
        axes[1, 1].set_title('Learning Rate Effect on Convergence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Mathematical formulas explanation
        print("\n📝 Mathematical Formulas:")
        print("-" * 30)
        print("1. Sigmoid Function: σ(z) = 1 / (1 + e^(-z))")
        print("2. Linear Combination: z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ")
        print("3. Hypothesis: h(x) = σ(θᵀx)")
        print("4. Cost Function: J(θ) = -(1/m) × Σ[y×log(h(x)) + (1-y)×log(1-h(x))]")
        print("5. Gradient: ∇J(θ) = (1/m) × Xᵀ × (h(x) - y)")
        print("6. Parameter Update: θ := θ - α × ∇J(θ)")
    
    def prepare_binary_classification(self):
        """Prepare data for binary classification (Setosa vs Others)."""
        print("\n🔧 Preparing Binary Classification Data...")
        print("=" * 50)
        
        # Create binary target (Setosa = 1, Others = 0)
        y_binary = (self.y == 0).astype(int)
        
        # Use only 2 features for better visualization (petal length and width)
        X_2d = self.X[:, [2, 3]]  # Petal length and width
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_2d, y_binary, test_size=0.3, random_state=42, stratify=y_binary
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Feature names: {[self.feature_names[2], self.feature_names[3]]}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test
    
    def train_custom_model(self, X_train, y_train, learning_rates=[0.01, 0.1, 1.0]):
        """Train custom logistic regression with different learning rates."""
        print("\n🎯 Training Custom Logistic Regression Models...")
        print("=" * 60)
        
        models = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Train models with different learning rates
        for i, lr in enumerate(learning_rates):
            print(f"\n--- Learning Rate: {lr} ---")
            model = CustomLogisticRegression(learning_rate=lr, max_iterations=1000)
            model.fit(X_train, y_train, verbose=False)
            models[lr] = model
            
            # Plot cost history
            if i < 2:
                axes[0, i].plot(model.cost_history, linewidth=2)
                axes[0, i].set_title(f'Cost History (LR = {lr})')
                axes[0, i].set_xlabel('Iterations')
                axes[0, i].set_ylabel('Cost')
                axes[0, i].grid(True, alpha=0.3)
        
        # Compare all learning rates
        axes[1, 0].set_title('Cost Comparison: Different Learning Rates')
        for lr in learning_rates:
            axes[1, 0].plot(models[lr].cost_history, label=f'LR = {lr}', linewidth=2)
        axes[1, 0].set_xlabel('Iterations')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Weight evolution for best model
        best_model = models[0.1]  # Usually 0.1 works well
        weight_history = np.array(best_model.weight_history)
        
        axes[1, 1].set_title('Weight Evolution During Training')
        axes[1, 1].plot(weight_history[:, 0], label='Weight 1 (Petal Length)', linewidth=2)
        axes[1, 1].plot(weight_history[:, 1], label='Weight 2 (Petal Width)', linewidth=2)
        axes[1, 1].set_xlabel('Iterations')
        axes[1, 1].set_ylabel('Weight Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return models
    
    def visualize_decision_boundary(self, model, X_train, y_train, X_test, y_test, scaler):
        """Visualize decision boundary and predictions."""
        print("\n🎨 Visualizing Decision Boundary...")
        print("=" * 50)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Training data with decision boundary
        h = 0.02
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_proba(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        axes[0].contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
        axes[0].contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        
        # Plot training points
        scatter = axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                                cmap='RdYlBu', edgecolors='black', s=50)
        axes[0].set_title('Training Data with Decision Boundary')
        axes[0].set_xlabel('Petal Length (scaled)')
        axes[0].set_ylabel('Petal Width (scaled)')
        
        # 2. Test predictions
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)
        
        scatter = axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, 
                                cmap='RdYlBu', edgecolors='black', s=50)
        axes[1].set_title('Test Predictions')
        axes[1].set_xlabel('Petal Length (scaled)')
        axes[1].set_ylabel('Petal Width (scaled)')
        
        # 3. Prediction probabilities
        scatter = axes[2].scatter(X_test[:, 0], X_test[:, 1], c=y_proba_test, 
                                cmap='RdYlBu', edgecolors='black', s=50)
        plt.colorbar(scatter, ax=axes[2])
        axes[2].set_title('Prediction Probabilities')
        axes[2].set_xlabel('Petal Length (scaled)')
        axes[2].set_ylabel('Petal Width (scaled)')
        
        plt.tight_layout()
        plt.show()
        
        # Print mathematical interpretation
        print(f"\n🔍 Model Parameters:")
        print(f"Weight 1 (Petal Length): {model.weights[0]:.4f}")
        print(f"Weight 2 (Petal Width): {model.weights[1]:.4f}")
        print(f"Bias: {model.bias:.4f}")
        
        print(f"\n📐 Decision Boundary Equation:")
        print(f"{model.weights[0]:.4f} × (Petal Length) + {model.weights[1]:.4f} × (Petal Width) + {model.bias:.4f} = 0")
        
        # Evaluate model
        accuracy = np.mean(y_pred_test == y_test)
        print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return y_pred_test, y_proba_test
    
    def step_by_step_prediction(self, model, X_test, y_test):
        """Show step-by-step prediction process."""
        print("\n🔍 Step-by-Step Prediction Process...")
        print("=" * 60)
        
        # Take first 5 test samples
        n_samples = min(5, len(X_test))
        
        print("For each test sample, we'll show:")
        print("1. Input features")
        print("2. Linear combination (z)")
        print("3. Sigmoid transformation")
        print("4. Final prediction")
        print()
        
        for i in range(n_samples):
            x = X_test[i]
            y_true = y_test[i]
            
            # Step 1: Linear combination
            z = np.dot(x, model.weights) + model.bias
            
            # Step 2: Sigmoid
            probability = model.sigmoid(z)
            
            # Step 3: Prediction
            prediction = 1 if probability >= 0.5 else 0
            
            print(f"Sample {i+1}:")
            print(f"  Features: [{x[0]:.3f}, {x[1]:.3f}]")
            print(f"  z = {model.weights[0]:.3f}×{x[0]:.3f} + {model.weights[1]:.3f}×{x[1]:.3f} + {model.bias:.3f} = {z:.3f}")
            print(f"  σ(z) = 1/(1+e^(-{z:.3f})) = {probability:.3f}")
            print(f"  Prediction: {prediction} (threshold: 0.5)")
            print(f"  True label: {y_true}")
            print(f"  Correct: {'✅' if prediction == y_true else '❌'}")
            print()
    
    def compare_with_sklearn(self, X_train, X_test, y_train, y_test):
        """Compare custom implementation with sklearn."""
        print("\n⚖️ Comparing Custom vs Scikit-learn Implementation...")
        print("=" * 60)
        
        from sklearn.linear_model import LogisticRegression
        
        # Train sklearn model
        sklearn_model = LogisticRegression(random_state=42)
        sklearn_model.fit(X_train, y_train)
        
        # Train custom model
        custom_model = CustomLogisticRegression(learning_rate=0.1, max_iterations=1000)
        custom_model.fit(X_train, y_train, verbose=False)
        
        # Compare predictions
        sklearn_pred = sklearn_model.predict(X_test)
        custom_pred = custom_model.predict(X_test)
        
        sklearn_accuracy = np.mean(sklearn_pred == y_test)
        custom_accuracy = np.mean(custom_pred == y_test)
        
        print(f"Scikit-learn Accuracy: {sklearn_accuracy:.4f}")
        print(f"Custom Implementation Accuracy: {custom_accuracy:.4f}")
        print(f"Difference: {abs(sklearn_accuracy - custom_accuracy):.4f}")
        
        # Compare parameters
        print(f"\nParameter Comparison:")
        print(f"Scikit-learn weights: {sklearn_model.coef_[0]}")
        print(f"Custom weights: {custom_model.weights}")
        print(f"Scikit-learn bias: {sklearn_model.intercept_[0]:.4f}")
        print(f"Custom bias: {custom_model.bias:.4f}")
        
        return custom_model, sklearn_model
    
    def run_complete_tutorial(self):
        """Run the complete custom logistic regression tutorial."""
        print("🌸 CUSTOM LOGISTIC REGRESSION TUTORIAL")
        print("=" * 70)
        print("Learn logistic regression by implementing it from scratch!")
        print("We'll build the mathematical foundation step by step.\n")
        
        # Step 1: Visualize mathematical functions
        self.visualize_mathematical_functions()
        
        # Step 2: Prepare data
        X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = self.prepare_binary_classification()
        
        # Step 3: Train custom models
        models = self.train_custom_model(X_train, y_train)
        
        # Step 4: Use the best model for further analysis
        best_model = models[0.1]  # Usually works well
        
        # Step 5: Visualize decision boundary
        y_pred, y_proba = self.visualize_decision_boundary(best_model, X_train, y_train, X_test, y_test, scaler)
        
        # Step 6: Step-by-step predictions
        self.step_by_step_prediction(best_model, X_test, y_test)
        
        # Step 7: Compare with sklearn
        custom_model, sklearn_model = self.compare_with_sklearn(X_train, X_test, y_train, y_test)
        
        print("\n🎉 Custom Logistic Regression Tutorial Complete!")
        print("=" * 60)
        print("You've successfully:")
        print("✅ Understood the mathematical foundation")
        print("✅ Implemented logistic regression from scratch")
        print("✅ Visualized the training process")
        print("✅ Analyzed decision boundaries")
        print("✅ Compared with scikit-learn")
        print("\nYou now understand how logistic regression works under the hood! 🚀")


def main():
    """Main function to run the tutorial."""
    tutorial = IrisCustomLogisticRegression()
    tutorial.run_complete_tutorial()


if __name__ == "__main__":
    main()
