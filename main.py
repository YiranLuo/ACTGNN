import numpy as np
import torch
import torch.nn as nn
import generate_data
import MLP_model
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from mpl_toolkits.mplot3d import Axes3D
from lshashpy3 import LSHash

def lsh_features(coor_array, hash_size, num_hash_tables):
    """
    Apply LSH to the coordinate array to generate hash features for nodes.
    :param coor_array: Numpy array of coordinates.
    :param num_features: Number of features for LSH.
    :param num_hash_tables: Number of hash tables for LSH.
    :return: Transformed feature array.
    """
    lsh = LSHash(hash_size=hash_size, input_dim=coor_array.shape[1], num_hashtables=num_hash_tables)

    # Adding items to LSH
    for idx, coor in enumerate(coor_array):
        lsh.index(coor, extra_data=idx)

    # Feature Engineering
    features = []
    for node_feature in coor_array:
        # Query for nearest neighbors
        nn_results = lsh.query(node_feature, num_results=50, distance_func="euclidean")
        
        # Extract distances and indices
        distances = [result[1] for result in nn_results]
        indices = [result[0][1] for result in nn_results]
        
        # Calculate features based on nearest neighbors
        avg_distance = np.mean(distances)
        density = len(distances)
        variance = np.var(distances)
        
        # Append features for this node
        features.append([avg_distance, density, variance])

    # Convert to numpy array for further analysis
    # features_array = np.array(features)
    
    return np.array(features).astype(np.float32)

I = 100  # number of data points
J = 2   # number of dimension
max_K = 3   # number of clusters

dataset_size = 20000

# Generate data for training and testing
input_array, output_array = generate_data.generate_classification_data(I, J, max_K, size=dataset_size)

# Turn data type from double to float32
input_array = input_array.astype(np.float32)
output_array = output_array.astype(np.float32)
# output_array = output_array.astype(np.int8)

# Create input and output tensors from arrays
input_tensor = torch.tensor(input_array)
output_tensor = torch.tensor(output_array)
output_tensor = output_tensor.view(-1, 1)

train_proportion = 0.8

# Calculate number of samples for training and testing
num_samples = input_tensor.shape[0]
num_train_samples = int(train_proportion * num_samples)
num_test_samples = num_samples - num_train_samples

# Create a TensorDataset to hold both input and output data
dataset = TensorDataset(input_tensor, output_tensor)

# Split the dataset into training and testing sets
train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples])

batch_size = 64

# Generate DataLoaders for training and testing set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# train_loader = DataLoader(train_dataset, shuffle=True)
# test_loader = DataLoader(test_dataset, shuffle=False)

input_size = I * J
hidden_size1 = int(input_size / 2)
hidden_size2 = 16
output_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Create MLP model
model = MLP_model.MLP(input_size, hidden_size1, hidden_size2, output_size).to(device)

# Use MSE as loss function
# loss_function = nn.MSELoss()
loss_function = nn.BCEWithLogitsLoss()
# loss_function = nn.CrossEntropyLoss()

learning_rate = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 200
# num_epochs_list = [20, 50, 100, 150, 200]

# Learning rate scheduler
# scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Training

print(f"I={I}      J={J}     max_K={max_K}")
print(f"Dataset size: {dataset_size}")
print(f"learning_rate: {learning_rate}")
print(f"num_epochs: {num_epochs}")
print("Starting training...")

train_errors = []
eval_errors = []

def train():
    model.train()
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # print(targets.size())
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        # pred = torch.argmax(outputs, dim=1, keepdim=True)
        # targets = targets.long()
        # targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=2).to(torch.float32)
        
        # Compute the loss
        loss = loss_function(outputs, targets)
        
        loss.backward()
        optimizer.step()
        

def test(loader):
    model.eval()
    
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        # pred = torch.argmax(outputs, dim=1, keepdim=True)
        pred = (outputs > 0.5).float()
        
        # correct += int((pred == targets).sum())
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    return correct / total

for epoch in range(num_epochs):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    
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