"""
Understanding w^T x + b: Linear Combination Visualization
========================================================

This tutorial demonstrates how the linear combination w^T x + b works in machine learning:
1. Simple house price prediction dataset (square footage, bedrooms → price)
2. Weight and bias initialization
3. Step-by-step visualization of how weights and bias update during training
4. Understanding the geometric interpretation of w^T x + b
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LinearCombinationVisualizer:
    """
    Visualizes how w^T x + b works in linear regression.
    """
    
    def __init__(self):
        """Initialize with a simple house price dataset."""
        print("🏠 Creating Simple House Price Dataset")
        print("=" * 50)
        
        # Create a simple synthetic dataset
        np.random.seed(42)
        n_samples = 20
        
        # Features: [square_footage, num_bedrooms]
        square_footage = np.random.uniform(800, 3000, n_samples)
        num_bedrooms = np.random.randint(1, 6, n_samples)
        
        # True relationship: price = 150 * sqft + 10000 * bedrooms + 50000 + noise
        true_w1, true_w2, true_b = 150, 10000, 50000
        noise = np.random.normal(0, 15000, n_samples)
        price = true_w1 * square_footage + true_w2 * num_bedrooms + true_b + noise
        
        # Store data
        self.X_original = np.column_stack([square_footage, num_bedrooms])
        self.y_original = price
        
        # Create DataFrame for easy viewing
        self.df = pd.DataFrame({
            'square_footage': square_footage,
            'num_bedrooms': num_bedrooms,
            'price': price
        })
        
        # Standardize features for better training
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X_original)
        self.y = (price - price.mean()) / price.std()  # Standardize target too
        
        print(f"Dataset created: {n_samples} houses")
        print(f"Features: Square Footage, Number of Bedrooms")
        print(f"Target: House Price")
        print(f"\nFirst 5 samples:")
        print(self.df.head())
        
        print(f"\nTrue relationship (before standardization):")
        print(f"Price = {true_w1} × sqft + {true_w2} × bedrooms + {true_b}")
    
    def visualize_dataset(self):
        """Visualize the dataset and the linear combination concept."""
        print("\n📊 Visualizing Dataset and Linear Combination Concept")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Original data scatter plots
        axes[0, 0].scatter(self.df['square_footage'], self.df['price'], alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Square Footage')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].set_title('Price vs Square Footage')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].scatter(self.df['num_bedrooms'], self.df['price'], alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Number of Bedrooms')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].set_title('Price vs Number of Bedrooms')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2. 3D visualization of the relationship
        ax_3d = fig.add_subplot(2, 3, 3, projection='3d')
        ax_3d.scatter(self.df['square_footage'], self.df['num_bedrooms'], self.df['price'], 
                     c=self.df['price'], cmap='viridis', alpha=0.7)
        ax_3d.set_xlabel('Square Footage')
        ax_3d.set_ylabel('Bedrooms')
        ax_3d.set_zlabel('Price ($)')
        ax_3d.set_title('3D View: Price vs Features')
        
        # 3. Standardized data
        axes[1, 0].scatter(self.X[:, 0], self.y, alpha=0.7, color='blue')
        axes[1, 0].set_xlabel('Square Footage (standardized)')
        axes[1, 0].set_ylabel('Price (standardized)')
        axes[1, 0].set_title('Standardized: Price vs Square Footage')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(self.X[:, 1], self.y, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Bedrooms (standardized)')
        axes[1, 1].set_ylabel('Price (standardized)')
        axes[1, 1].set_title('Standardized: Price vs Bedrooms')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 4. Linear combination visualization concept
        x_demo = np.array([1.5, 0.8])  # Example feature vector
        w_demo = np.array([0.7, 1.2])  # Example weight vector
        b_demo = 0.3
        
        axes[1, 2].arrow(0, 0, x_demo[0], x_demo[1], head_width=0.1, head_length=0.1, 
                        fc='blue', ec='blue', label='x (features)')
        axes[1, 2].arrow(0, 0, w_demo[0], w_demo[1], head_width=0.1, head_length=0.1, 
                        fc='red', ec='red', label='w (weights)')
        
        # Show dot product geometrically
        dot_product = np.dot(x_demo, w_demo)
        axes[1, 2].text(0.5, 1.5, f'w^T x = {dot_product:.2f}', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
        axes[1, 2].text(0.5, 1.2, f'w^T x + b = {dot_product + b_demo:.2f}', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        axes[1, 2].set_xlim(-0.2, 2)
        axes[1, 2].set_ylim(-0.2, 2)
        axes[1, 2].set_xlabel('Feature 1')
        axes[1, 2].set_ylabel('Feature 2')
        axes[1, 2].set_title('Linear Combination: w^T x + b')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_linear_combination_step_by_step(self):
        """Show step-by-step how w^T x + b is calculated."""
        print("\n🔍 Step-by-Step Linear Combination Calculation")
        print("=" * 60)
        
        # Take first 3 samples
        for i in range(3):
            x = self.X[i]
            y_true = self.y[i]
            x_orig = self.X_original[i]
            y_orig = self.y_original[i]
            
            print(f"\n📋 Sample {i+1}:")
            print(f"   Original features: [{x_orig[0]:.0f} sqft, {x_orig[1]} bedrooms]")
            print(f"   Standardized features: x = [{x[0]:.3f}, {x[1]:.3f}]")
            print(f"   True price: ${y_orig:,.0f} (standardized: {y_true:.3f})")
            
            # Example weights
            w = np.array([0.5, 0.8])
            b = 0.2
            
            print(f"\n   🧮 Linear Combination Calculation:")
            print(f"   Given weights: w = [{w[0]:.3f}, {w[1]:.3f}]")
            print(f"   Given bias: b = {b:.3f}")
            print(f"   ")
            print(f"   Step 1: w^T x = w₁×x₁ + w₂×x₂")
            print(f"           w^T x = {w[0]:.3f}×{x[0]:.3f} + {w[1]:.3f}×{x[1]:.3f}")
            print(f"           w^T x = {w[0]*x[0]:.3f} + {w[1]*x[1]:.3f}")
            print(f"           w^T x = {np.dot(w, x):.3f}")
            print(f"   ")
            print(f"   Step 2: Add bias")
            print(f"           y_pred = w^T x + b")
            print(f"           y_pred = {np.dot(w, x):.3f} + {b:.3f}")
            print(f"           y_pred = {np.dot(w, x) + b:.3f}")
            
            prediction = np.dot(w, x) + b
            error = y_true - prediction
            print(f"   ")
            print(f"   📊 Result:")
            print(f"   Predicted: {prediction:.3f}")
            print(f"   Actual: {y_true:.3f}")
            print(f"   Error: {error:.3f}")
    
    def visualize_weight_bias_training(self):
        """Visualize how weights and bias are updated during training."""
        print("\n🎯 Training Visualization: Weight and Bias Updates")
        print("=" * 60)
        
        # Simple linear regression implementation
        class SimpleLinearRegression:
            def __init__(self, learning_rate=0.01):
                self.learning_rate = learning_rate
                self.weights = None
                self.bias = None
                self.weight_history = []
                self.bias_history = []
                self.cost_history = []
            
            def fit(self, X, y, epochs=50):
                # Initialize parameters
                n_features = X.shape[1]
                self.weights = np.random.normal(0, 0.1, n_features)  # Small random values
                self.bias = 0.0  # Start with zero bias
                
                print(f"🎲 Initial Parameters:")
                print(f"   Weights: {self.weights}")
                print(f"   Bias: {self.bias:.4f}")
                print()
                
                m = len(X)
                
                for epoch in range(epochs):
                    # Forward pass: compute predictions
                    predictions = np.dot(X, self.weights) + self.bias
                    
                    # Compute cost (Mean Squared Error)
                    cost = np.mean((predictions - y) ** 2)
                    
                    # Compute gradients
                    dw = (2/m) * np.dot(X.T, (predictions - y))
                    db = (2/m) * np.sum(predictions - y)
                    
                    # Update parameters
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
                    
                    # Store history
                    self.weight_history.append(self.weights.copy())
                    self.bias_history.append(self.bias)
                    self.cost_history.append(cost)
                    
                    # Print progress
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch:2d}: Cost = {cost:.4f}, "
                              f"Weights = [{self.weights[0]:.4f}, {self.weights[1]:.4f}], "
                              f"Bias = {self.bias:.4f}")
                
                print(f"\n✅ Final Parameters:")
                print(f"   Weights: [{self.weights[0]:.4f}, {self.weights[1]:.4f}]")
                print(f"   Bias: {self.bias:.4f}")
        
        # Train the model
        model = SimpleLinearRegression(learning_rate=0.1)
        model.fit(self.X, self.y, epochs=50)
        
        # Visualize the training process
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Cost function over time
        axes[0, 0].plot(model.cost_history, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Cost (MSE)')
        axes[0, 0].set_title('Cost Function During Training')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Weight evolution
        weight_history = np.array(model.weight_history)
        axes[0, 1].plot(weight_history[:, 0], 'r-', linewidth=2, label='Weight 1 (sqft)')
        axes[0, 1].plot(weight_history[:, 1], 'g-', linewidth=2, label='Weight 2 (bedrooms)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Weight Value')
        axes[0, 1].set_title('Weight Evolution During Training')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Bias evolution
        axes[0, 2].plot(model.bias_history, 'purple', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Bias Value')
        axes[0, 2].set_title('Bias Evolution During Training')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Weight trajectory in 2D space
        axes[1, 0].plot(weight_history[:, 0], weight_history[:, 1], 'o-', alpha=0.7)
        axes[1, 0].scatter(weight_history[0, 0], weight_history[0, 1], 
                          color='red', s=100, label='Start', zorder=5)
        axes[1, 0].scatter(weight_history[-1, 0], weight_history[-1, 1], 
                          color='green', s=100, label='End', zorder=5)
        axes[1, 0].set_xlabel('Weight 1 (sqft)')
        axes[1, 0].set_ylabel('Weight 2 (bedrooms)')
        axes[1, 0].set_title('Weight Trajectory in Parameter Space')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Predictions vs actual
        final_predictions = np.dot(self.X, model.weights) + model.bias
        axes[1, 1].scatter(self.y, final_predictions, alpha=0.7)
        axes[1, 1].plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 1].set_xlabel('Actual Price (standardized)')
        axes[1, 1].set_ylabel('Predicted Price (standardized)')
        axes[1, 1].set_title('Final Predictions vs Actual')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Feature importance visualization
        feature_names = ['Square Footage', 'Bedrooms']
        axes[1, 2].bar(feature_names, np.abs(model.weights), color=['blue', 'green'], alpha=0.7)
        axes[1, 2].set_ylabel('|Weight| (Importance)')
        axes[1, 2].set_title('Feature Importance (|Weights|)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return model
    
    def demonstrate_prediction_process(self, model):
        """Show how predictions are made with the trained model."""
        print("\n🔮 Making Predictions with Trained Model")
        print("=" * 50)
        
        # Make predictions on new data
        new_houses = np.array([
            [2000, 3],  # 2000 sqft, 3 bedrooms
            [1500, 2],  # 1500 sqft, 2 bedrooms
            [2500, 4]   # 2500 sqft, 4 bedrooms
        ])
        
        # Standardize new data
        new_houses_scaled = self.scaler.transform(new_houses)
        
        print("🏠 Predicting prices for new houses:")
        print("=" * 40)
        
        for i, (house_orig, house_scaled) in enumerate(zip(new_houses, new_houses_scaled)):
            print(f"\nHouse {i+1}: {house_orig[0]:.0f} sqft, {house_orig[1]} bedrooms")
            print(f"Standardized: [{house_scaled[0]:.3f}, {house_scaled[1]:.3f}]")
            
            # Step-by-step prediction
            w1, w2 = model.weights
            b = model.bias
            x1, x2 = house_scaled
            
            print(f"\n🧮 Prediction Calculation:")
            print(f"   w^T x + b = w₁×x₁ + w₂×x₂ + b")
            print(f"   w^T x + b = {w1:.4f}×{x1:.3f} + {w2:.4f}×{x2:.3f} + {b:.4f}")
            print(f"   w^T x + b = {w1*x1:.4f} + {w2*x2:.4f} + {b:.4f}")
            
            prediction_scaled = np.dot(house_scaled, model.weights) + model.bias
            print(f"   w^T x + b = {prediction_scaled:.4f}")
            
            # Convert back to original scale
            prediction_orig = prediction_scaled * self.y_original.std() + self.y_original.mean()
            print(f"\n💰 Predicted Price: ${prediction_orig:,.0f}")
    
    def visualize_geometric_interpretation(self):
        """Show the geometric interpretation of w^T x + b."""
        print("\n📐 Geometric Interpretation of w^T x + b")
        print("=" * 50)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Vector interpretation
        x_example = np.array([1.2, 0.8])
        w_example = np.array([0.6, 1.0])
        
        axes[0].arrow(0, 0, x_example[0], x_example[1], head_width=0.05, head_length=0.05,
                     fc='blue', ec='blue', linewidth=2, label='x (features)')
        axes[0].arrow(0, 0, w_example[0], w_example[1], head_width=0.05, head_length=0.05,
                     fc='red', ec='red', linewidth=2, label='w (weights)')
        
        # Show projection
        dot_product = np.dot(x_example, w_example)
        w_norm = np.linalg.norm(w_example)
        projection = (dot_product / w_norm) * (w_example / w_norm)
        
        axes[0].arrow(0, 0, projection[0], projection[1], head_width=0.05, head_length=0.05,
                     fc='green', ec='green', linewidth=2, linestyle='--', label='projection')
        
        axes[0].text(0.5, 1.2, f'w^T x = {dot_product:.2f}', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
        
        axes[0].set_xlim(-0.2, 1.5)
        axes[0].set_ylim(-0.2, 1.5)
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].set_title('Vector Interpretation: w^T x')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Hyperplane visualization (2D case)
        x_range = np.linspace(-2, 2, 100)
        # For w^T x + b = 0, we have w1*x1 + w2*x2 + b = 0
        # Solving for x2: x2 = -(w1*x1 + b) / w2
        w1, w2, b = 0.5, 0.8, 0.3
        if abs(w2) > 1e-10:
            x2_line = -(w1 * x_range + b) / w2
            axes[1].plot(x_range, x2_line, 'r-', linewidth=2, label=f'{w1:.1f}x₁ + {w2:.1f}x₂ + {b:.1f} = 0')
        
        # Show some points and their values
        test_points = np.array([[-1, -1], [0, 0], [1, 1], [-1, 1]])
        for point in test_points:
            value = w1 * point[0] + w2 * point[1] + b
            color = 'blue' if value > 0 else 'red'
            axes[1].scatter(point[0], point[1], c=color, s=100, alpha=0.7)
            axes[1].text(point[0]+0.1, point[1]+0.1, f'{value:.1f}', fontsize=10)
        
        axes[1].set_xlim(-2, 2)
        axes[1].set_ylim(-2, 2)
        axes[1].set_xlabel('x₁')
        axes[1].set_ylabel('x₂')
        axes[1].set_title('Decision Boundary: w^T x + b = 0')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Effect of bias
        x_vals = np.linspace(-2, 2, 100)
        w_fixed = 0.5
        biases = [-1, 0, 1]
        
        for bias in biases:
            y_vals = w_fixed * x_vals + bias
            axes[2].plot(x_vals, y_vals, linewidth=2, label=f'y = {w_fixed}x + {bias}')
        
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y = wx + b')
        axes[2].set_title('Effect of Bias on Linear Function')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_tutorial(self):
        """Run the complete linear combination tutorial."""
        print("🏠 LINEAR COMBINATION TUTORIAL: w^T x + b")
        print("=" * 70)
        print("Understanding how weights and bias work in machine learning!")
        print("Using house price prediction as an example.\n")
        
        # Step 1: Visualize dataset
        self.visualize_dataset()
        
        # Step 2: Demonstrate linear combination step by step
        self.demonstrate_linear_combination_step_by_step()
        
        # Step 3: Visualize training process
        model = self.visualize_weight_bias_training()
        
        # Step 4: Show prediction process
        self.demonstrate_prediction_process(model)
        
        # Step 5: Geometric interpretation
        self.visualize_geometric_interpretation()
        
        print("\n🎉 Linear Combination Tutorial Complete!")
        print("=" * 60)
        print("You've learned:")
        print("✅ How w^T x + b is calculated step by step")
        print("✅ How weights and bias are initialized and updated")
        print("✅ The geometric interpretation of linear combinations")
        print("✅ How to make predictions with trained parameters")
        print("✅ The effect of each component on the final output")
        print("\nNow you understand the foundation of all linear models! 🚀")


def main():
    """Main function to run the tutorial."""
    tutorial = LinearCombinationVisualizer()
    tutorial.run_complete_tutorial()


if __name__ == "__main__":
    main()
