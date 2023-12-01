import numpy as np
import torch
import torch.nn as nn
import generate_data
import MLP_model
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from mpl_toolkits.mplot3d import Axes3D

I = 50  # number of data points
J = 4   # number of dimension
K = 2   # number of clusters

dataset_size = 500000

# Generate data for training and testing
input_array, output_array = generate_data.generate_data_all(I, J, K, size=dataset_size)

# Turn data type from double to float32
input_array = input_array.astype(np.float32)
output_array = output_array.astype(np.float32)

# Create input and output tensors from arrays
input_tensor = torch.tensor(input_array)
output_tensor = torch.tensor(output_array)

train_proportion = 0.8

# Calculate number of samples for training and testing
num_samples = input_tensor.shape[0]
num_train_samples = int(train_proportion * num_samples)
num_test_samples = num_samples - num_train_samples

# Create a TensorDataset to hold both input and output data
dataset = TensorDataset(input_tensor, output_tensor)

# Split the dataset into training and testing sets
train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples])

batch_size = 32

# Generate DataLoaders for training and testing set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = I * J
hidden_size = 64
output_size = J * K

# Create MLP model
model = MLP_model.MLP(input_size, hidden_size, output_size)

# Use MSE as loss function
loss_function = nn.MSELoss()

learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 200
# num_epochs_list = [20, 50, 100, 150, 200]

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Training

print(f"I={I}      J={J}     K={K}")
print(f"Dataset size: {dataset_size}")
print(f"learning_rate: {learning_rate}")
print(f"num_epochs: {num_epochs}")
print("Starting training...")

train_errors = []
eval_errors = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute the loss
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Update the learning rate
    scheduler.step()        
    train_error = total_loss / len(train_loader)
    train_errors.append(train_error)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_error}")
    
    # Testing
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()
            
        avg_loss = total_loss / len(test_loader)
        eval_errors.append(avg_loss)
        print(f"         Test Loss: {avg_loss}")
    
# # The following code are used for visualization for one data point
# # Generate one date point
# data_matrix, centroids_matrix = generate_data.generate_data_matrix_and_centroids_matrix(I, J, K)

# data_matrix = data_matrix.astype(np.float32)
# centroids_matrix = centroids_matrix.astype(np.float32)

# # Plot the points as blue dots, and the centroids as red dots
# x_coords = data_matrix[:, 0]
# y_coords = data_matrix[:, 1]
# z_coords = data_matrix[:, 2]

# x_coords_center = centroids_matrix[:, 0]
# y_coords_center = centroids_matrix[:, 1]
# z_coords_center = centroids_matrix[:, 2]

# # Use the trained model to predict the centroids
# flattened_data_matrix = data_matrix.reshape(1, -1)
# input_tensor = torch.tensor(flattened_data_matrix)
# output_tensor = model(input_tensor)

# reshaped_output_tensor = output_tensor.view(K, J)
# centroids_matrix_pred = reshaped_output_tensor.detach().numpy()

# # Plot the predicted centroids as orange dots
# x_coords_center_pred = centroids_matrix_pred[:, 0]
# y_coords_center_pred = centroids_matrix_pred[:, 1]
# z_coords_center_pred = centroids_matrix_pred[:, 2]

# Create a figure with subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

fig = plt.figure(figsize=(12, 8))

# Plot the data visualization on the first subplot (ax1)
# ax1.scatter(x_coords, y_coords, color='blue', marker='o')
# ax1.scatter(x_coords_center, y_coords_center, color='red', marker='o')
# ax1.scatter(x_coords_center_pred, y_coords_center_pred, color='orange', marker='o')

# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(x_coords, y_coords, z_coords, color='blue', marker='o', label='Data Points')
# ax1.scatter(x_coords_center, y_coords_center, z_coords_center, color='red', marker='o', label='Original Centroids')
# ax1.scatter(x_coords_center_pred, y_coords_center_pred, z_coords_center_pred, color='orange', marker='o', label='Predicted Centroids')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')
# ax1.legend()

# Plot training and evaluation errors against epochs on the second subplot (ax2)
ax2 = fig.add_subplot(122)
ax2.plot(range(1, num_epochs + 1), train_errors, label='Training Error')
ax2.plot(range(1, num_epochs + 1), eval_errors, label='Evaluation Error')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Error')
ax2.legend()

plt.show()