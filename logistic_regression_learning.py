import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Generate a simple squiggly line dataset
def generate_data():
    np.random.seed(42)
    # Create x values
    x = np.linspace(-5, 5, 10)
    
    # Create a squiggly line with some noise
    y = np.sin(x) + 0.2 * np.random.randn(len(x))
    
    # Create binary labels based on whether points are above or below the line
    labels = (y > 0).astype(int)
    
    return x, y, labels

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)

def compute_loss(X, y, weights):
    m = len(y)
    predictions = predict(X, weights)
    loss = (-1/m) * (np.dot(y, np.log(predictions)) + np.dot(1-y, np.log(1-predictions)))
    return loss

def train_logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    m = len(y)
    # Add bias term to X
    X = np.column_stack((np.ones(m), X))
    
    # Initialize weights
    weights = np.zeros(X.shape[1])
    
    # Store weights and loss history for visualization
    weights_history = [weights.copy()]
    loss_history = []
    
    for _ in range(num_iterations):
        predictions = predict(X, weights)
        error = predictions - y
        
        # Update weights
        weights -= (learning_rate/m) * np.dot(X.T, error)
        
        # Store weights and loss
        weights_history.append(weights.copy())
        loss_history.append(compute_loss(X, y, weights))
    
    return weights, weights_history, loss_history

def visualize_learning_process(x, y, labels, weights_history, loss_history):
    # Create a figure with 3 subplots
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Initial and final decision boundaries
    plt.subplot(2, 2, 1)
    plt.scatter(x[labels==0], y[labels==0], color='red', label='Class 0', alpha=0.7)
    plt.scatter(x[labels==1], y[labels==1], color='blue', label='Class 1', alpha=0.7)
    
    # Create x values for plotting decision boundaries
    x_plot = np.linspace(-5, 5, 100)
    
    # Plot initial decision boundary (first weights)
    initial_weights = weights_history[0]
    if len(initial_weights) >= 3 and abs(initial_weights[2]) > 1e-6:
        y_initial = (-initial_weights[0] - initial_weights[1] * x_plot) / initial_weights[2]
        plt.plot(x_plot, y_initial, 'g--', label='Initial Boundary', alpha=0.5)
    
    # Plot final decision boundary (last weights)
    final_weights = weights_history[-1]
    if len(final_weights) >= 3 and abs(final_weights[2]) > 1e-6:
        y_final = (-final_weights[0] - final_weights[1] * x_plot) / final_weights[2]
        plt.plot(x_plot, y_final, 'g-', label='Final Boundary', lw=2)
    
    plt.axhline(y=0, color='black', linestyle=':', alpha=0.3, label='True Boundary')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Decision Boundaries')
    plt.legend()
    
    # Subplot 2: Loss curve
    plt.subplot(2, 2, 2)
    plt.plot(loss_history, 'b-')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Weights evolution
    plt.subplot(2, 1, 2)
    weights_array = np.array(weights_history)
    for i in range(weights_array.shape[1]):
        plt.plot(weights_array[:, i], label=f'w{i}')
    plt.xlabel('Iteration')
    plt.ylabel('Weight Value')
    plt.title('Weights Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_gradient_descent():
    # Simple quadratic function: f(w) = w² + 5
    def f(w):
        return w**2 + 5
    
    # Derivative: f'(w) = 2w
    def df(w):
        return 2*w
    
    # Gradient descent parameters
    w = 6.0  # Starting point
    learning_rate = 0.2
    num_steps = 5
    
    # Store values for plotting
    w_history = [w]
    f_history = [f(w)]
    
    print("Starting Gradient Descent Steps:")
    print(f"Initial w = {w:.4f}, f(w) = {f(w):.4f}")
    
    # Run gradient descent
    for i in range(num_steps):
        # Calculate gradient
        grad = df(w)
        
        # Update w
        w = w - learning_rate * grad
        
        # Store values
        w_history.append(w)
        f_history.append(f(w))
        
        print(f"Step {i+1}: w = {w:.4f}, f(w) = {f(w):.4f}, gradient = {grad:.4f}")
    
    # Plot the function and steps
    plt.figure(figsize=(10, 6))
    w_vals = np.linspace(-7, 7, 100)
    plt.plot(w_vals, f(w_vals), 'b-', label='f(w) = w² + 5')
    plt.scatter(w_history, f_history, c='red', s=100, zorder=5)
    
    # Draw arrows and show slopes
    for i in range(len(w_history)-1):
        # Draw the step arrow
        plt.annotate('', xy=(w_history[i+1], f_history[i+1]), 
                    xytext=(w_history[i], f_history[i]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Calculate and display the slope at this point
        slope = df(w_history[i])
        plt.text(w_history[i], f_history[i] + 2, 
                f'slope={slope:.2f}', 
                ha='center', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.xlabel('Weight (w)')
    plt.ylabel('Loss f(w)')
    plt.title('Gradient Descent Steps')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # First show the simple gradient descent visualization
    visualize_gradient_descent()
    
    # Then run the logistic regression example
    print("\nNow running logistic regression example:")
    x, y, labels = generate_data()
    X = np.column_stack((x, y))
    weights, weights_history, loss_history = train_logistic_regression(
        X, labels, learning_rate=0.1, num_iterations=200)
    
    print(f"\nFinal weights: {weights}")
    print(f"Final loss: {loss_history[-1]}")
    
    # visualize_learning_process(x, y, labels, weights_history, loss_history)
