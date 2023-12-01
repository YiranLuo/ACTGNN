import numpy as np
import matplotlib.pyplot as plt
import random

# Function to generate centroids matrix M
def generate_centroids_matrix(K, J):
    centroids_matrix = np.random.rand(K, J)
    centroids_matrix = centroids_matrix * 10
    return centroids_matrix

# Function to generate assignments matrix A
def generate_assignments_matrix(I, K):
    random_indices = np.random.choice(K, size=I)
    assignment_matrix = np.zeros((I, K))
    assignment_matrix[np.arange(I), random_indices] = 1
    return assignment_matrix

# Function to generate the approximation of data matrix X_approx
def generate_data_matrix_approx(I, J, K):
    centroids_matrix = generate_centroids_matrix(K=K, J=J)
    assignments_matrix = generate_assignments_matrix(I=I, K=K)
    return np.dot(assignments_matrix, centroids_matrix), centroids_matrix
    
# Function to generate both data matrix and centroids matrix
def generate_data_matrix_and_centroids_matrix(I, J, K, noise_mean=0.0, noise_std_dev=0.1):
    data_matrix_approx, centroids_matrix = generate_data_matrix_approx(I=I, J=J, K=K)
    
    # Add Gaussian noise
    noise = np.random.normal(noise_mean, noise_std_dev, size=data_matrix_approx.shape)
    return data_matrix_approx + noise, centroids_matrix

# Function to generate data for binary classification task (to determine if K = 0 or K > 0)
def generate_classification_data(I, J, max_K, size):
    input = []
    output = []
    
    for i in range(size):
        K = random.randint(0, max_K) # both included
        if K == 0:
            data_matrix = np.random.rand(I, J)
            data_matrix = data_matrix * 10

            # x_coords = data_matrix[:, 0]
            # y_coords = data_matrix[:, 1]

            # plt.scatter(x_coords, y_coords)
            # plt.legend()
            # plt.show()

            # print(f"K = {K}")
            # print(data_matrix)
        else:
            data_matrix, centroids_matrix = generate_data_matrix_and_centroids_matrix(I, J, K)
            # print(f"K = {K}")
            # print(data_matrix)
            # x_coords = data_matrix[:, 0]
            # y_coords = data_matrix[:, 1]

            # plt.scatter(x_coords, y_coords)
            # plt.legend()
            # plt.show()
        
        input.append(data_matrix.flatten())
        output.append(0 if K == 0 else 1) # 0 for K=0, 1 for K>0
        
    input_array = np.array(input)
    output_array = np.array(output)

    return input_array, output_array

    
# Function to generate synthetic data
def generate_data_all(I, J, K, size):
    input = []
    output = []

    # fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i in range(size):
        data_matrix, centroids_matrix = generate_data_matrix_and_centroids_matrix(I, J, K)
        # TO DO: Add different variance to the same set of matrix

        data_matrix_flattened = data_matrix.flatten()
        centroids_matrix_flattened = centroids_matrix.flatten()

        input.append(data_matrix_flattened)
        output.append(centroids_matrix_flattened)

        # if i < 4:
        #     row = i // 2
        #     col = i % 2
        #     ax = axes[row, col]

            
        #     # Plot the points as blue dots, and the centroids as red dots
        #     x_coords = data_matrix[:, 0]
        #     y_coords = data_matrix[:, 1]

        #     x_coords_center = centroids_matrix[:, 0]
        #     y_coords_center = centroids_matrix[:, 1]

        #     # Plot the data visualization on the first subplot (ax1)
        #     ax.scatter(x_coords, y_coords, color='blue', marker='o')
        #     ax.scatter(x_coords_center, y_coords_center, color='red', marker='o')
        #     plt.scatter(x_coords, y_coords, color='blue', marker='o')
        #     plt.scatter(x_coords_center, y_coords_center, color='red', marker='o')
        #     # ax.scatter(x_coords_center_pred, y_coords_center_pred, color='orange', marker='o')
    

    input_array = np.array(input)
    output_array = np.array(output)
    
    plt.show()

    return input_array, output_array