import numpy as np
import matplotlib.pyplot as plt
import random

def add_noise(data, noise_type='gaussian', **params):
    """
    Add different types of noise to the data matrix.
    
    Parameters:
    - data: Input data matrix
    - noise_type: Type of noise to add ('gaussian', 'uniform', 'salt_pepper', 'poisson', 'multiplicative')
    - params: Additional parameters for specific noise types
    """
    if noise_type == 'gaussian':
        mean = params.get('mean', 0.0)
        std = params.get('std', 0.1)
        noise = np.random.normal(mean, std, size=data.shape)
        return data + noise
    
    elif noise_type == 'uniform':
        low = params.get('low', -0.1)
        high = params.get('high', 0.1)
        noise = np.random.uniform(low, high, size=data.shape)
        return data + noise
    
    elif noise_type == 'salt_pepper':
        prob = params.get('prob', 0.05)
        noisy = np.copy(data)
        # Salt noise
        salt = np.random.random(data.shape) < prob/2
        noisy[salt] = 1.0
        # Pepper noise
        pepper = np.random.random(data.shape) < prob/2
        noisy[pepper] = 0.0
        return noisy
    
    elif noise_type == 'poisson':
        return np.random.poisson(data)
    
    elif noise_type == 'multiplicative':
        mean = params.get('mean', 1.0)
        std = params.get('std', 0.1)
        noise = np.random.normal(mean, std, size=data.shape)
        return data * noise
    
    return data  # Return original data if noise_type not recognized

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
    
def generate_data_matrix_and_centroids_matrix(I, J, K, noise_type='gaussian', **noise_params):
    data_matrix_approx, centroids_matrix = generate_data_matrix_approx(I=I, J=J, K=K)
    noisy_data = add_noise(data_matrix_approx, noise_type, **noise_params)
    return noisy_data, centroids_matrix

# def generate_classification_data(I, J, max_K, size):
#     input = []
#     output = []
    
#     for i in range(size):
#         K = random.randint(0, max_K) # both included
#         if K == 0:
#             data_matrix = np.random.rand(I, J)
#             data_matrix = data_matrix * 10
#         else:
#             data_matrix, centroids_matrix = generate_data_matrix_and_centroids_matrix(I, J, K)
        
#         input.append(data_matrix)
#         output.append(0 if K <= 1 else 1) # 0 for K=0, 1 for K>0
        
#     input_array = np.array(input)
#     output_array = np.array(output)

#     return input_array, output_array

def generate_data_gnn(I, J, max_K, size):
    coor=[]
    label=[]

    for i in range(size):
        K = random.randint(0, max_K) # both included
        if K == 0:
            data_matrix = np.random.rand(I, J)
            data_matrix = data_matrix * 10
        else:
            data_matrix, centroids_matrix = generate_data_matrix_and_centroids_matrix(I, J, K)
        
        coor.append(data_matrix)
        label.append(0 if K <= 1 else 1) # 0 for K=0, 1 for K>0
        
    coor_array = np.squeeze(np.array(coor))
    label_array = np.squeeze(np.array(label))

    return coor_array, label_array
    
def generate_data_gnn_balanced(I, J, max_K, size, noise_type='gaussian', **noise_params):
    coor = []
    label = []

    # Calculate the number of samples for each label
    num_zero_label = size // 2
    num_nonzero_label = size - num_zero_label

    # Generate data for K = 0
    for _ in range(num_zero_label):
        data_matrix = np.random.rand(I, J) * 10
        data_matrix = add_noise(data_matrix, noise_type, **noise_params)
        coor.append(data_matrix)
        label.append(0)

    # Generate data for K > 0
    for _ in range(num_nonzero_label):
        K = random.randint(1, max_K)  # Ensure K > 0
        data_matrix, centroids_matrix = generate_data_matrix_and_centroids_matrix(I, J, K, noise_type, **noise_params)
        coor.append(data_matrix)
        label.append(1)

    # Combine and shuffle the data and labels together
    combined = list(zip(coor, label))
    random.shuffle(combined)
    coor, label = zip(*combined)

    coor_array = np.squeeze(np.array(coor))
    label_array = np.squeeze(np.array(label))

    return coor_array, label_array

def generate_data_gnn_balanced_varying_I(min_I, max_I, J, max_K, size, noise_type='gaussian', **noise_params):
    coor = []
    label = []
    I_values = []  # Store the I value used for each sample

    # Calculate the number of samples for each label
    num_zero_label = size // 2
    num_nonzero_label = size - num_zero_label

    # Generate data for K = 0
    for _ in range(num_zero_label):
        I = random.randint(min_I, max_I)  # Random I for each sample
        data_matrix = np.random.rand(I, J) * 10
        data_matrix = add_noise(data_matrix, noise_type, **noise_params)
        coor.append(data_matrix)
        label.append(0)
        I_values.append(I)

    # Generate data for K > 0
    for _ in range(num_nonzero_label):
        I = random.randint(min_I, max_I)  # Random I for each sample
        K = random.randint(1, max_K)  # Ensure K > 0
        data_matrix, centroids_matrix = generate_data_matrix_and_centroids_matrix(I, J, K, noise_type, **noise_params)
        coor.append(data_matrix)
        label.append(1)
        I_values.append(I)

    # Combine and shuffle the data and labels together
    combined = list(zip(coor, label, I_values))
    random.shuffle(combined)
    coor, label, I_values = zip(*combined)

    # Convert to numpy array with dtype=object to handle different shapes
    coor_array = np.array(coor, dtype=object)
    label_array = np.array(label)
    
    return coor_array, label_array, np.array(I_values)

if __name__ == "__main__":
    # Test parameters
    I = 200  # number of points
    J = 2    # dimensions
    max_K = 100  # max number of clusters
    size = 2  # number of samples to generate
    
    # Test different noise types
    noise_types = ['gaussian', 'uniform', 'salt_pepper', 'multiplicative']
    noise_params = {
        'gaussian': {'mean': 0.0, 'std': 0.4},
        'uniform': {'low': -0.3, 'high': 0.3},
        'salt_pepper': {'prob': 0.1},
        'multiplicative': {'mean': 1.0, 'std': 0.2}
    }
    
    # Create subplots for each noise type
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, noise_type in enumerate(noise_types):
        # Generate data with specific noise
        coor_array, label_array = generate_data_gnn_balanced(
            I, J, max_K, size, 
            noise_type=noise_type, 
            **noise_params[noise_type]
        )
        
        # Plot results
        ax = axes[idx]
        for i in range(len(coor_array)):
            color = 'blue' if label_array[i] == 0 else 'red'
            label = 'No Clusters' if label_array[i] == 0 else 'Has Clusters'
            ax.scatter(coor_array[i][:, 0], coor_array[i][:, 1], 
                      c=color, alpha=0.6, label=label)
        
        ax.set_title(f'Noise Type: {noise_type}')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()