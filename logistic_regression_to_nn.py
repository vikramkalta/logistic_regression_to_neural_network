import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, 2:4] # petal length and width
y = (iris.target != 0).astype(int).reshape(-1, 1) # binary classification
# label = 0 if Setosa, 1 if Versicolor or Virginica (binary task)

# To make it simpler, let's just keep two classes
X = X[y.flatten() != 2]
y = y[y.flatten() != 2]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialise weights
W = np.random.randn(2, 1) # 2 features
b = 0

# Training loop
lr = 0.1
for epoch in range(10):
    # Forward pass
    z = np.dot(X_train, W) + b
    y_hat = sigmoid(z)

    # Loss (binary cross-entropy)
    loss = -np.mean(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat))

    # Backprop (gradients)
    dz = y_hat - y_train
    dw = np.dot(X_train.T, dz) / len(X_train)
    db = np.mean(dz)

    # Update
    W -= lr * dw
    b -= lr * db

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Predictions
y_pred = sigmoid(np.dot(X_test, W) + b)
y_pred_class = (y_pred > 0.5).astype(int)
accuracy = np.mean(y_pred_class == y_test)
print(f"Accuracy: {accuracy}")
    
# Tiny neural network: 2 -> 3 -> 1
hidden_size = 3

# Initialise weights
W1 = np.random.randn(2, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, 1)
b2 = 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
lr = 0.1
for epoch in range(10):
    # Forward pass
    z1 = np.dot(X_train, W1) + b1 # input -> hidden
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2 # hidden -> output
    y_hat = sigmoid(z2)

    # ---- Loss ----
    loss = -np.mean(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat))

    # Backprop
    dz2 = y_hat - y_train
    dW2 = np.dot(a1.T, dz2) / len(X_train)
    db2 = np.mean(dz2)

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * a1 * (1 - a1)
    dW1 = np.dot(X_train.T, dz1) / len(X_train)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Predictions
a1_test = sigmoid(np.dot(X_test, W1) + b1)
y_pred = sigmoid(np.dot(a1_test, W2) + b2)
y_pred_class = (y_pred > 0.5).astype(int)
accuracy = np.mean(y_pred_class == y_test)
print(f"Accuracy: {accuracy}")
    
    
    
    

    