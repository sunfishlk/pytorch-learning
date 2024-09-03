import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def swish(x):
    return x * sigmoid(x)

# Generate x values
x = np.linspace(-10, 10, 400)

# Compute y values for each activation function
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_elu = elu(x)
y_swish = swish(x)

# Plot each activation function
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.plot(x, y_sigmoid, label='Sigmoid', color='blue')
plt.title('Sigmoid Function')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(x, y_tanh, label='Tanh', color='orange')
plt.title('Tanh Function')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(x, y_relu, label='ReLU', color='green')
plt.title('ReLU Function')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(x, y_leaky_relu, label='Leaky ReLU', color='red')
plt.title('Leaky ReLU Function')
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(x, y_elu, label='ELU', color='purple')
plt.title('ELU Function')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(x, y_swish, label='Swish', color='brown')
plt.title('Swish Function')
plt.grid(True)

plt.tight_layout()
plt.show()
