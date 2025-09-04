"""
Gradient Descent Step-by-Step Visualization
==========================================

This visualizes exactly what happens in each step of gradient descent:
1. Forward pass: predictions = w^T x + b
2. Cost calculation: MSE = mean((predictions - y)^2)
3. Gradient computation: dw, db
4. Parameter updates: w = w - α * dw, b = b - α * db
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GradientDescentVisualizer:
    """
    Visualizes the gradient descent process step by step.
    """
    
    def __init__(self):
        """Create a simple dataset for visualization."""
        print("🎯 Creating Simple Dataset for Gradient Descent Visualization")
        print("=" * 70)
        
        # Create simple 2D dataset
        np.random.seed(42)
        n_samples = 8  # Small dataset for clarity
        
        # Features: [square_footage, num_bedrooms]
        square_footage = np.array([1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800])
        num_bedrooms = np.array([2, 2, 3, 3, 4, 4, 5, 5])
        
        # Simple linear relationship: price = 200 * sqft + 15000 * bedrooms + 50000
        price = 200 * square_footage + 15000 * num_bedrooms + 50000 + np.random.normal(0, 5000, n_samples)
        
        # Store original data
        self.X_original = np.column_stack([square_footage, num_bedrooms])
        self.y_original = price
        
        # Standardize for better visualization
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        self.X = scaler_X.fit_transform(self.X_original)
        self.y = scaler_y.fit_transform(self.y_original.reshape(-1, 1)).flatten()
        
        # Create DataFrame for display
        self.df = pd.DataFrame({
            'sqft': square_footage,
            'bedrooms': num_bedrooms,
            'price': price
        })
        
        print(f"Dataset: {n_samples} houses")
        print("\nOriginal Data:")
        print(self.df)
        
        print(f"\nStandardized Features (X):")
        for i, (orig, std) in enumerate(zip(self.X_original, self.X)):
            print(f"House {i+1}: [{orig[0]:4.0f}, {orig[1]}] → [{std[0]:6.3f}, {std[1]:6.3f}]")
        
        print(f"\nStandardized Prices (y): {self.y.round(3)}")
    
    def visualize_single_gradient_step(self, weights, bias, learning_rate=0.1, step_num=0):
        """
        Visualize a single step of gradient descent in detail.
        """
        print(f"\n🔍 GRADIENT DESCENT STEP {step_num}")
        print("=" * 60)
        
        m = len(self.X)
        
        print(f"📊 Current Parameters:")
        print(f"   Weights: w = [{weights[0]:7.4f}, {weights[1]:7.4f}]")
        print(f"   Bias: b = {bias:7.4f}")
        print(f"   Learning Rate: α = {learning_rate}")
        
        # Step 1: Forward Pass - Compute Predictions
        print(f"\n🚀 Step 1: Forward Pass (Compute Predictions)")
        print(f"   predictions = X @ w + b")
        print(f"   For each sample i: pred[i] = w₁×x₁[i] + w₂×x₂[i] + b")
        
        predictions = np.dot(self.X, weights) + bias
        
        print(f"\n   Sample-by-sample calculation:")
        for i in range(m):
            x1, x2 = self.X[i]
            pred = weights[0] * x1 + weights[1] * x2 + bias
            print(f"   House {i+1}: {weights[0]:6.3f}×{x1:6.3f} + {weights[1]:6.3f}×{x2:6.3f} + {bias:6.3f} = {pred:7.4f}")
        
        print(f"\n   📋 Predictions: {predictions.round(4)}")
        print(f"   📋 Actual:      {self.y.round(4)}")
        
        # Step 2: Compute Cost (Mean Squared Error)
        print(f"\n💰 Step 2: Compute Cost (Mean Squared Error)")
        print(f"   cost = (1/m) × Σ(predictions - actual)²")
        
        errors = predictions - self.y
        squared_errors = errors ** 2
        cost = np.mean(squared_errors)
        
        print(f"\n   Error calculation:")
        for i in range(m):
            print(f"   House {i+1}: error = {predictions[i]:7.4f} - {self.y[i]:7.4f} = {errors[i]:7.4f}, error² = {squared_errors[i]:7.4f}")
        
        print(f"\n   📊 Total Cost = (1/{m}) × {squared_errors.sum():.4f} = {cost:.6f}")
        
        # Step 3: Compute Gradients
        print(f"\n📈 Step 3: Compute Gradients")
        print(f"   dw = (2/m) × X^T @ (predictions - actual)")
        print(f"   db = (2/m) × Σ(predictions - actual)")
        
        # Weight gradients
        dw = (2/m) * np.dot(self.X.T, errors)
        db = (2/m) * np.sum(errors)
        
        print(f"\n   Weight gradient calculation:")
        print(f"   dw₁ = (2/{m}) × Σ(error[i] × x₁[i])")
        print(f"   dw₂ = (2/{m}) × Σ(error[i] × x₂[i])")
        
        # Show detailed gradient calculation
        dw1_terms = errors * self.X[:, 0]
        dw2_terms = errors * self.X[:, 1]
        
        print(f"\n   Sample contributions to dw₁:")
        for i in range(m):
            print(f"   House {i+1}: {errors[i]:7.4f} × {self.X[i,0]:6.3f} = {dw1_terms[i]:7.4f}")
        print(f"   Sum = {dw1_terms.sum():.4f}, dw₁ = (2/{m}) × {dw1_terms.sum():.4f} = {dw[0]:.6f}")
        
        print(f"\n   Sample contributions to dw₂:")
        for i in range(m):
            print(f"   House {i+1}: {errors[i]:7.4f} × {self.X[i,1]:6.3f} = {dw2_terms[i]:7.4f}")
        print(f"   Sum = {dw2_terms.sum():.4f}, dw₂ = (2/{m}) × {dw2_terms.sum():.4f} = {dw[1]:.6f}")
        
        print(f"\n   Bias gradient:")
        print(f"   db = (2/{m}) × Σ(error[i]) = (2/{m}) × {errors.sum():.4f} = {db:.6f}")
        
        print(f"\n   📊 Gradients: dw = [{dw[0]:7.4f}, {dw[1]:7.4f}], db = {db:7.4f}")
        
        # Step 4: Update Parameters
        print(f"\n🔄 Step 4: Update Parameters")
        print(f"   w_new = w_old - α × dw")
        print(f"   b_new = b_old - α × db")
        
        new_weights = weights - learning_rate * dw
        new_bias = bias - learning_rate * db
        
        print(f"\n   Weight updates:")
        print(f"   w₁_new = {weights[0]:7.4f} - {learning_rate} × {dw[0]:7.4f} = {weights[0]:7.4f} - {learning_rate * dw[0]:7.4f} = {new_weights[0]:7.4f}")
        print(f"   w₂_new = {weights[1]:7.4f} - {learning_rate} × {dw[1]:7.4f} = {weights[1]:7.4f} - {learning_rate * dw[1]:7.4f} = {new_weights[1]:7.4f}")
        
        print(f"\n   Bias update:")
        print(f"   b_new = {bias:7.4f} - {learning_rate} × {db:7.4f} = {bias:7.4f} - {learning_rate * db:7.4f} = {new_bias:7.4f}")
        
        print(f"\n   📊 Updated Parameters:")
        print(f"   New Weights: [{new_weights[0]:7.4f}, {new_weights[1]:7.4f}]")
        print(f"   New Bias: {new_bias:7.4f}")
        
        # Return updated parameters and metrics
        return new_weights, new_bias, cost, predictions, errors, dw, db
    
    def visualize_gradient_descent_process(self, num_steps=5):
        """
        Visualize multiple steps of gradient descent.
        """
        print(f"\n🎯 COMPLETE GRADIENT DESCENT PROCESS")
        print("=" * 70)
        
        # Initialize parameters
        weights = np.array([0.1, -0.2])  # Start with some initial values
        bias = 0.0
        learning_rate = 0.3
        
        # Storage for visualization
        weight_history = [weights.copy()]
        bias_history = [bias]
        cost_history = []
        
        print(f"🎲 Initial Parameters:")
        print(f"   Weights: [{weights[0]:7.4f}, {weights[1]:7.4f}]")
        print(f"   Bias: {bias:7.4f}")
        
        # Perform gradient descent steps
        for step in range(num_steps):
            weights, bias, cost, predictions, errors, dw, db = self.visualize_single_gradient_step(
                weights, bias, learning_rate, step + 1
            )
            
            # Store history
            weight_history.append(weights.copy())
            bias_history.append(bias)
            cost_history.append(cost)
            
            print(f"\n   ✅ Step {step + 1} Complete - Cost: {cost:.6f}")
            
            if step < num_steps - 1:
                print(f"\n" + "─" * 70)
        
        # Create comprehensive visualization
        self.create_gradient_descent_plots(weight_history, bias_history, cost_history, learning_rate)
        
        return weight_history, bias_history, cost_history
    
    def create_gradient_descent_plots(self, weight_history, bias_history, cost_history, learning_rate):
        """
        Create comprehensive plots showing the gradient descent process.
        """
        print(f"\n📊 Creating Gradient Descent Visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Cost function over iterations
        axes[0, 0].plot(cost_history, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Cost (MSE)')
        axes[0, 0].set_title('Cost Function During Training')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add cost values as text
        for i, cost in enumerate(cost_history):
            axes[0, 0].text(i, cost, f'{cost:.3f}', ha='center', va='bottom')
        
        # 2. Weight evolution
        weight_history = np.array(weight_history)
        iterations = range(len(weight_history))
        
        axes[0, 1].plot(iterations, weight_history[:, 0], 'ro-', linewidth=2, markersize=8, label='w₁ (sqft)')
        axes[0, 1].plot(iterations, weight_history[:, 1], 'go-', linewidth=2, markersize=8, label='w₂ (bedrooms)')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Weight Value')
        axes[0, 1].set_title('Weight Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Bias evolution
        axes[0, 2].plot(bias_history, 'mo-', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Bias Value')
        axes[0, 2].set_title('Bias Evolution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add bias values as text
        for i, bias in enumerate(bias_history):
            axes[0, 2].text(i, bias, f'{bias:.3f}', ha='center', va='bottom')
        
        # 4. Weight trajectory in 2D parameter space
        axes[1, 0].plot(weight_history[:, 0], weight_history[:, 1], 'o-', linewidth=2, markersize=8, alpha=0.7)
        axes[1, 0].scatter(weight_history[0, 0], weight_history[0, 1], color='red', s=150, label='Start', zorder=5)
        axes[1, 0].scatter(weight_history[-1, 0], weight_history[-1, 1], color='green', s=150, label='End', zorder=5)
        
        # Add step numbers
        for i, (w1, w2) in enumerate(weight_history):
            axes[1, 0].text(w1, w2, str(i), ha='center', va='center', fontweight='bold')
        
        axes[1, 0].set_xlabel('Weight 1 (sqft)')
        axes[1, 0].set_ylabel('Weight 2 (bedrooms)')
        axes[1, 0].set_title('Weight Trajectory in Parameter Space')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Predictions vs Actual (final model)
        final_weights = weight_history[-1]
        final_bias = bias_history[-1]
        final_predictions = np.dot(self.X, final_weights) + final_bias
        
        axes[1, 1].scatter(self.y, final_predictions, s=100, alpha=0.7, color='blue')
        axes[1, 1].plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 1].set_xlabel('Actual Price (standardized)')
        axes[1, 1].set_ylabel('Predicted Price (standardized)')
        axes[1, 1].set_title('Final Predictions vs Actual')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Residuals (errors) over iterations
        # Calculate predictions for each iteration
        residuals_by_iteration = []
        for i, (w, b) in enumerate(zip(weight_history[1:], bias_history[1:])):
            preds = np.dot(self.X, w) + b
            residuals = preds - self.y
            residuals_by_iteration.append(residuals)
        
        if residuals_by_iteration:
            residuals_array = np.array(residuals_by_iteration)
            for house_idx in range(len(self.X)):
                axes[1, 2].plot(range(1, len(weight_history)), residuals_array[:, house_idx], 
                               'o-', alpha=0.7, label=f'House {house_idx+1}')
            axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 2].set_xlabel('Iteration')
            axes[1, 2].set_ylabel('Residual (Prediction - Actual)')
            axes[1, 2].set_title('Residuals Over Training')
            axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Learning rate sensitivity analysis
        learning_rates = [0.01, 0.1, 0.3, 0.5]
        axes[2, 0].set_title('Learning Rate Sensitivity')
        
        for lr in learning_rates:
            # Quick simulation for different learning rates
            w_temp = np.array([0.1, -0.2])
            b_temp = 0.0
            costs_temp = []
            
            for _ in range(10):
                preds = np.dot(self.X, w_temp) + b_temp
                cost = np.mean((preds - self.y) ** 2)
                costs_temp.append(cost)
                
                errors = preds - self.y
                dw = (2/len(self.X)) * np.dot(self.X.T, errors)
                db = (2/len(self.X)) * np.sum(errors)
                
                w_temp -= lr * dw
                b_temp -= lr * db
            
            axes[2, 0].plot(costs_temp, label=f'LR = {lr}', linewidth=2)
        
        axes[2, 0].set_xlabel('Iteration')
        axes[2, 0].set_ylabel('Cost')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Feature importance (final weights)
        feature_names = ['Square Footage', 'Bedrooms']
        axes[2, 1].bar(feature_names, np.abs(final_weights), color=['blue', 'green'], alpha=0.7)
        axes[2, 1].set_ylabel('|Weight| (Importance)')
        axes[2, 1].set_title('Final Feature Importance')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Add weight values on bars
        for i, weight in enumerate(final_weights):
            axes[2, 1].text(i, abs(weight), f'{weight:.3f}', ha='center', va='bottom')
        
        # 9. Summary statistics
        axes[2, 2].axis('off')
        summary_text = f"""
        GRADIENT DESCENT SUMMARY
        ========================
        
        Initial Cost: {cost_history[0]:.6f}
        Final Cost: {cost_history[-1]:.6f}
        Cost Reduction: {((cost_history[0] - cost_history[-1]) / cost_history[0] * 100):.2f}%
        
        Initial Weights: [{weight_history[0][0]:.4f}, {weight_history[0][1]:.4f}]
        Final Weights: [{final_weights[0]:.4f}, {final_weights[1]:.4f}]
        
        Initial Bias: {bias_history[0]:.4f}
        Final Bias: {final_bias:.4f}
        
        Learning Rate: {learning_rate}
        Iterations: {len(cost_history)}
        
        Final R² Score: {1 - (np.sum((final_predictions - self.y)**2) / np.sum((self.y - np.mean(self.y))**2)):.4f}
        """
        
        axes[2, 2].text(0.1, 0.9, summary_text, transform=axes[2, 2].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_gradient_intuition(self):
        """
        Show the intuition behind gradients and why they work.
        """
        print(f"\n🧠 Understanding Gradient Intuition")
        print("=" * 50)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Simple 1D cost function
        w_range = np.linspace(-2, 2, 100)
        # Simple quadratic cost function: (w - 1)^2
        cost_1d = (w_range - 1) ** 2
        gradient_1d = 2 * (w_range - 1)
        
        axes[0, 0].plot(w_range, cost_1d, 'b-', linewidth=2, label='Cost Function')
        
        # Show gradient at a few points
        points = [-1, 0, 0.5, 1.5]
        for point in points:
            cost_val = (point - 1) ** 2
            grad_val = 2 * (point - 1)
            
            # Draw gradient arrow
            axes[0, 0].arrow(point, cost_val, -0.3 * np.sign(grad_val), 0, 
                           head_width=0.1, head_length=0.1, fc='red', ec='red')
            axes[0, 0].text(point, cost_val + 0.2, f'∇={grad_val:.1f}', ha='center')
        
        axes[0, 0].scatter([1], [0], color='green', s=100, label='Minimum', zorder=5)
        axes[0, 0].set_xlabel('Weight')
        axes[0, 0].set_ylabel('Cost')
        axes[0, 0].set_title('1D Cost Function and Gradients')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Gradient descent path
        w_path = [1.8]  # Starting point
        lr = 0.3
        
        for _ in range(10):
            current_w = w_path[-1]
            gradient = 2 * (current_w - 1)
            new_w = current_w - lr * gradient
            w_path.append(new_w)
        
        cost_path = [(w - 1) ** 2 for w in w_path]
        
        axes[0, 1].plot(w_range, cost_1d, 'b-', linewidth=2, alpha=0.5)
        axes[0, 1].plot(w_path, cost_path, 'ro-', linewidth=2, markersize=8, label='Gradient Descent Path')
        axes[0, 1].scatter([w_path[0]], [cost_path[0]], color='red', s=150, label='Start', zorder=5)
        axes[0, 1].scatter([w_path[-1]], [cost_path[-1]], color='green', s=150, label='End', zorder=5)
        
        axes[0, 1].set_xlabel('Weight')
        axes[0, 1].set_ylabel('Cost')
        axes[0, 1].set_title('Gradient Descent Optimization Path')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Learning rate effects
        learning_rates = [0.1, 0.5, 1.0, 1.5]
        
        for lr in learning_rates:
            w_temp = 1.8
            w_trajectory = [w_temp]
            
            for _ in range(15):
                gradient = 2 * (w_temp - 1)
                w_temp = w_temp - lr * gradient
                w_trajectory.append(w_temp)
                
                if abs(w_temp - 1) < 0.001:  # Converged
                    break
            
            cost_trajectory = [(w - 1) ** 2 for w in w_trajectory]
            axes[1, 0].plot(w_trajectory, cost_trajectory, 'o-', label=f'LR = {lr}', alpha=0.8)
        
        axes[1, 0].plot(w_range, cost_1d, 'k-', linewidth=1, alpha=0.3)
        axes[1, 0].set_xlabel('Weight')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].set_title('Effect of Different Learning Rates')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Why gradients point to steepest ascent
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X_mesh, Y_mesh = np.meshgrid(x, y)
        
        # 2D cost function: (x-1)^2 + (y+0.5)^2
        Z = (X_mesh - 1)**2 + (Y_mesh + 0.5)**2
        
        # Gradients
        dZ_dx = 2 * (X_mesh - 1)
        dZ_dy = 2 * (Y_mesh + 0.5)
        
        contour = axes[1, 1].contour(X_mesh, Y_mesh, Z, levels=10, alpha=0.6)
        axes[1, 1].clabel(contour, inline=True, fontsize=8)
        
        # Show gradient vectors (pointing uphill)
        step = 3
        axes[1, 1].quiver(X_mesh[::step, ::step], Y_mesh[::step, ::step], 
                         dZ_dx[::step, ::step], dZ_dy[::step, ::step], 
                         alpha=0.6, color='red', scale=20)
        
        axes[1, 1].scatter([1], [-0.5], color='green', s=100, label='Minimum', zorder=5)
        axes[1, 1].set_xlabel('Weight 1')
        axes[1, 1].set_ylabel('Weight 2')
        axes[1, 1].set_title('2D Cost Surface and Gradients')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n💡 Key Insights:")
        print(f"   • Gradients point in the direction of STEEPEST INCREASE")
        print(f"   • We move OPPOSITE to the gradient (negative gradient)")
        print(f"   • Learning rate controls how big steps we take")
        print(f"   • Too high learning rate → oscillation or divergence")
        print(f"   • Too low learning rate → slow convergence")
    
    def run_complete_visualization(self):
        """Run the complete gradient descent visualization."""
        print("🎯 GRADIENT DESCENT STEP-BY-STEP VISUALIZATION")
        print("=" * 80)
        print("Understanding exactly what happens in each training iteration!")
        print()
        
        # Step 1: Show the dataset
        print("Dataset Overview:")
        print(self.df)
        
        # Step 2: Demonstrate gradient intuition
        # self.demonstrate_gradient_intuition()
        
        # Step 3: Run detailed gradient descent
        weight_history, bias_history, cost_history = self.visualize_gradient_descent_process(num_steps=5)
        
        print(f"\n🎉 Gradient Descent Visualization Complete!")
        print("=" * 60)
        print("You now understand:")
        print("✅ How predictions are computed: w^T x + b")
        print("✅ How cost (MSE) is calculated")
        print("✅ How gradients are computed")
        print("✅ How parameters are updated")
        print("✅ Why gradient descent works")
        print("✅ The effect of learning rate")
        print("\nYou've mastered the core of machine learning optimization! 🚀")


def main():
    """Main function to run the visualization."""
    visualizer = GradientDescentVisualizer()
    visualizer.run_complete_visualization()


if __name__ == "__main__":
    main()
