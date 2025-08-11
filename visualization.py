import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_squiggly_data(n_points=100):
    """
    Create a squiggly dataset with two classes that aren't easily separable by a straight line.
    """
    np.random.seed(42)  # For reproducibility
    
    # Create a squiggly pattern
    x = np.linspace(-3, 3, n_points)
    y = np.sin(x) + np.random.normal(0, 0.2, n_points)
    
    # Create two classes based on the squiggly pattern
    labels = np.zeros(n_points)
    labels[y > 0] = 1
    
    # Add some noise to make it more interesting
    x += np.random.normal(0, 0.1, n_points)
    y += np.random.normal(0, 0.1, n_points)
    
    return np.column_stack((x, y)), labels

def plot_dataset(X, y, title="Dataset"):
    """
    Plot the dataset with different colors for each class.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Class 0', alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    """
    Plot the decision boundary of the logistic regression model.
    """
    plt.figure(figsize=(8, 6))
    
    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                        np.linspace(y_min, y_max, 200))
    
    # Get predictions for the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour
    plt.contourf(xx, yy, Z, alpha=0.3)
    
    # Plot the data points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Class 0', alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', alpha=0.7)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
