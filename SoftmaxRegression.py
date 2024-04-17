import torch
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Generate a synthetic dataset
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# Convert the dataset to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset for Softmax Regression')
plt.show()

# Model parameters
weights = torch.randn((2, 3), requires_grad=True)  # 2 features, 3 classes
bias = torch.zeros(3, requires_grad=True)

def model(x):
    return torch.softmax(x @ weights + bias, dim=1)

# Example prediction
preds = model(X_tensor)
print(preds[:5])  # Display the first 5 predictions

def cross_entropy_loss(predictions, targets):
    log_p = -torch.log(predictions[range(targets.shape[0]), targets])
    return torch.mean(log_p)

# Learning rate
lr = 0.1
# Number of epochs
epochs = 100

# Training loop
for epoch in range(epochs):
    preds = model(X_tensor)
    loss = cross_entropy_loss(preds, y_tensor)
    loss.backward()
    
    with torch.no_grad():
        weights -= lr * weights.grad
        bias -= lr * bias.grad
        weights.grad.zero_()
        bias.grad.zero_()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Define a function to predict the class and visualize it
def predict_and_visualize(new_point):
    # Convert the new point to a PyTorch tensor
    new_point_tensor = torch.tensor(new_point, dtype=torch.float32).unsqueeze(0)
    
    # Use the trained model to predict the class
    prediction = model(new_point_tensor)
    predicted_class = torch.argmax(prediction).item()
    
    # Now let's plot the original dataset
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    
    # And plot the new point with the predicted class
    color_map = plt.cm.viridis
    class_colors = [color_map(i) for i in [0.0, 0.5, 1.0]]
    predicted_color = class_colors[predicted_class]
    
    plt.scatter(new_point[0], new_point[1], color=predicted_color, edgecolor='black', s=100, zorder=3)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Synthetic Dataset with Predicted Point')
    
    # Create a legend for the new point
    legend_patch = mpatches.Patch(color=predicted_color, label=f'Predicted class: {predicted_class}')
    plt.legend(handles=[legend_patch])
    
    plt.show()
    return predicted_class

# Example usage: Let's predict the class of a new point (1.0, -10.0)
new_data_point = [3.0, 5.0]
predicted_class = predict_and_visualize(new_data_point)
print(f"The predicted class for new data point {new_data_point} is {predicted_class}")

def plot_linear_discriminants(weights, bias):
    # Generate a grid of points to plot decision boundaries
    x_min, x_max = X_tensor[:, 0].min() - 1, X_tensor[:, 0].max() + 1
    y_min, y_max = X_tensor[:, 1].min() - 1, X_tensor[:, 1].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Flatten the grid to pass through the model
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    
    # Compute the class scores from the linear part (before softmax)
    scores = grid_tensor @ weights + bias.unsqueeze(0)
    
    # Compute the predicted class labels
    _, predicted_classes = torch.max(scores, 1)
    predicted_classes = predicted_classes.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, predicted_classes.numpy(), alpha=0.5, cmap='viridis')
    
    # Plot the original data points for reference
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linear Discriminant Functions')
    plt.show()

# Plot linear discriminants after training
plot_linear_discriminants(weights, bias)