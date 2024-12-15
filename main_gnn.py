import torch
import generate_data
import numpy as np
import os
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from lshashpy3 import LSHash

# Get device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

class ClusterDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None):
        self.data_list = data_list
        super(ClusterDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        if self.pre_transform is not None:
            self.data_list = [self.pre_transform(data) for data in self.data_list]
        
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])

def rbf_kernel(x1, x2, sigma):
    # Calculate on CPU
    distance_squared = torch.sum((x1 - x2) ** 2, dim=1)
    return torch.exp(-distance_squared / (2 * sigma ** 2))

def lsh_features(coor_array, hash_size, num_hash_tables):
    """
    Apply LSH to the coordinate array to generate hash features for nodes.
    :param coor_array: Numpy array of coordinates.
    :param num_features: Number of features for LSH.
    :param num_hash_tables: Number of hash tables for LSH.
    :return: Transformed feature array.
    """
    lsh = LSHash(hash_size=hash_size, input_dim=coor_array.shape[1], num_hashtables=num_hash_tables)

    # Calculate number of neighbors based on dataset size
    num_neighbors = min(int(0.25 * len(coor_array)), 100) # Use 25% of points up to max of 100

    # Adding items to LSH
    for idx, coor in enumerate(coor_array):
        lsh.index(coor, extra_data=idx)

    # Feature Engineering
    features = []
    for node_feature in coor_array:
        # Query for nearest neighbors
        nn_results = lsh.query(node_feature, num_results=num_neighbors, distance_func="euclidean")
        
        # Extract distances and indices
        distances = [result[1] for result in nn_results]
        indices = [result[0][1] for result in nn_results]
        
        # Calculate features based on nearest neighbors
        avg_distance = np.mean(distances)
        density = len(distances)
        variance = np.var(distances)
        
        # Append features for this node
        features.append([avg_distance, density, variance])

    return np.array(features).astype(np.float32)

def generate_graph_dataset(coor_array, label_array, KNN_dataset_path, similarity_kernel, sigma):
    print("Starting graph dataset generation...")
    
    # Handle varying-sized arrays individually
    dataset_size = len(coor_array)
    # Create coor and label tensors on CPU
    coor_tensor = [torch.tensor(arr.astype(np.float32)) for arr in coor_array]
    label_tensor = torch.tensor(label_array)

    graph_list = []
    
    assert similarity_kernel in ['euclidean', 'none', 'euclidean', 'cosine', 'rbf'], "Invalid similarity_kernel, choices: 'euclidean', 'none', 'euclidean', 'cosine', 'rbf'"

    print(f"Using {similarity_kernel} similarity kernel...")

    match similarity_kernel:
        case 'none':
            print("Creating graphs with no edge weights...")
            for i in range(dataset_size):
                if i % 100 == 0:
                    print(f"Processing graph {i}/{dataset_size}")
                num_nodes = coor_tensor[i].shape[0]
                k = min(int(0.6 * num_nodes), 99)
                edge_index = knn_graph(coor_tensor[i], k=k, loop=False)
                data = Data(pos=coor_tensor[i], y=label_tensor[i], x=coor_tensor[i], edge_index=edge_index)
                data.edge_index = to_undirected(data.edge_index)
                graph_list.append(data)
        
        case 'euclidean':
            print("Creating graphs with euclidean distances as edge weights...")
            for i in range(dataset_size):
                if i % 100 == 0:
                    print(f"Processing graph {i}/{dataset_size}")
                num_nodes = coor_tensor[i].shape[0]
                k = min(int(0.5 * num_nodes), 99)
                edge_index = knn_graph(coor_tensor[i], k=k, loop=False)

                # Compute distances for all edges
                start_nodes, end_nodes = edge_index
                distances = torch.norm(coor_tensor[i][start_nodes] - coor_tensor[i][end_nodes], dim=1)

                # Create a graph data object with edge weights
                data = Data(x=coor_tensor[i], pos=coor_tensor[i], edge_index=edge_index, y=label_tensor[i], edge_attr=distances)

                # Ensure the graph is undirected
                data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)

                graph_list.append(data)
                
        case 'cosine':
            print("Creating graphs with cosine similarities as edge weights...")
            for i in range(dataset_size):
                if i % 100 == 0:
                    print(f"Processing graph {i}/{dataset_size}")
                num_nodes = coor_tensor[i].shape[0]
                k = min(int(0.5 * num_nodes), 99)
                edge_index = knn_graph(coor_tensor[i], k=k, loop=False)

                # Retrieve the feature vectors of the start and end nodes of each edge
                start_vectors = coor_tensor[i][edge_index[0]]
                end_vectors = coor_tensor[i][edge_index[1]]

                # Calculate cosine similarity for each pair of vectors (edge)
                similarities = F.cosine_similarity(start_vectors, end_vectors, dim=1)

                # Create a graph data object with cosine similarities as edge attributes
                data = Data(x=coor_tensor[i], pos=coor_tensor[i], edge_index=edge_index, y=label_tensor[i], edge_attr=similarities)

                # Ensure the graph is undirected and edge attributes are correctly handled
                data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)

                graph_list.append(data)
        
        case 'rbf':
            print(f"Creating graphs with RBF kernel (sigma={sigma}) as edge weights...")
            for i in range(dataset_size):
                if i % 100 == 0:
                    print(f"Processing graph {i}/{dataset_size}")
                num_nodes = coor_tensor[i].shape[0]
                k = min(int(0.5 * num_nodes), 99)
                edge_index = knn_graph(coor_tensor[i], k=k, loop=False)

                # Retrieve the feature vectors of the start and end nodes of each edge
                start_nodes = edge_index[0]
                end_nodes = edge_index[1]
                start_vectors = coor_tensor[i][start_nodes]
                end_vectors = coor_tensor[i][end_nodes]

                # Calculate RBF kernel values for each edge
                rbf_values = rbf_kernel(start_vectors, end_vectors, sigma=sigma)

                # Create a graph data object with RBF kernel values as edge attributes
                data = Data(x=coor_tensor[i], pos=coor_tensor[i], edge_index=edge_index, y=label_tensor[i], edge_attr=rbf_values)

                # Ensure the graph is undirected
                data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)

                graph_list.append(data)
            
    print("Creating final dataset...")
    dataset = ClusterDataset(root=KNN_dataset_path, data_list=graph_list)
    print("Graph dataset generation complete!")
    return dataset

def generate_graph_dataset_lsh(coor_array, label_array, KNN_dataset_path, similarity_kernel, sigma, hash_size, num_hash_tables):
    print("Starting LSH graph dataset generation...")
    print(f"Using hash_size={hash_size}, num_hash_tables={num_hash_tables}")
    
    # Handle varying-sized arrays individually
    dataset_size = len(coor_array)
    # Create coor and label tensors on CPU
    coor_tensor = [torch.tensor(arr.astype(np.float32)) for arr in coor_array]
    label_tensor = torch.tensor(label_array)
    
    graph_list = []
    
    assert similarity_kernel in ['euclidean', 'none', 'euclidean', 'cosine', 'rbf'], "Invalid similarity_kernel, choices: 'euclidean', 'none', 'euclidean', 'cosine', 'rbf'"

    print(f"Using {similarity_kernel} similarity kernel...")

    match similarity_kernel:
        case 'none':
            print("Creating LSH graphs with no edge weights...")
            for i in range(dataset_size):
                if i % 100 == 0:
                    print(f"Processing graph {i}/{dataset_size}")
                lsh_coor_array = lsh_features(coor_array[i], hash_size=hash_size, num_hash_tables=num_hash_tables)
                lsh_coor_tensor = torch.tensor(lsh_coor_array)
                num_nodes = lsh_coor_tensor.shape[0]
                k = min(int(0.5 * num_nodes), 99)
                edge_index = knn_graph(coor_tensor[i], k=k, loop=False)
                data = Data(pos=lsh_coor_tensor, y=label_tensor[i], x=coor_tensor[i], edge_index=edge_index)
                data.edge_index = to_undirected(data.edge_index)
                graph_list.append(data)
        
        case 'euclidean':
            print("Creating LSH graphs with euclidean distances as edge weights...")
            for i in range(dataset_size):
                if i % 100 == 0:
                    print(f"Processing graph {i}/{dataset_size}")
                lsh_coor_array = lsh_features(coor_array[i], hash_size=hash_size, num_hash_tables=num_hash_tables)
                lsh_coor_tensor = torch.tensor(lsh_coor_array)
                num_nodes = lsh_coor_tensor.shape[0]
                k = min(int(0.5 * num_nodes), 99)
                edge_index = knn_graph(coor_tensor[i], k=k, loop=False)

                # Compute distances for all edges
                start_nodes, end_nodes = edge_index
                distances = torch.norm(coor_tensor[i][start_nodes] - coor_tensor[i][end_nodes], dim=1)

                # Create a graph data object with edge weights
                data = Data(x=lsh_coor_tensor, pos=coor_tensor[i], edge_index=edge_index, y=label_tensor[i], edge_attr=distances)

                # Ensure the graph is undirected
                data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)

                graph_list.append(data)
                
        case 'cosine':
            print("Creating LSH graphs with cosine similarities as edge weights...")
            for i in range(dataset_size):
                if i % 100 == 0:
                    print(f"Processing graph {i}/{dataset_size}")
                lsh_coor_array = lsh_features(coor_array[i], hash_size=hash_size, num_hash_tables=num_hash_tables)
                lsh_coor_tensor = torch.tensor(lsh_coor_array)
                num_nodes = lsh_coor_tensor.shape[0]
                k = min(int(0.5 * num_nodes), 99)
                edge_index = knn_graph(coor_tensor[i], k=k, loop=False)

                # Retrieve the feature vectors of the start and end nodes of each edge
                start_vectors = coor_tensor[i][edge_index[0]]
                end_vectors = coor_tensor[i][edge_index[1]]

                # Calculate cosine similarity for each pair of vectors (edge)
                similarities = F.cosine_similarity(start_vectors, end_vectors, dim=1)

                # Create a graph data object with cosine similarities as edge attributes
                data = Data(x=lsh_coor_tensor, pos=coor_tensor[i], edge_index=edge_index, y=label_tensor[i], edge_attr=similarities)

                # Ensure the graph is undirected and edge attributes are correctly handled
                data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)

                graph_list.append(data)
        
        case 'rbf':
            print(f"Creating LSH graphs with RBF kernel (sigma={sigma}) as edge weights...")
            for i in range(dataset_size):
                if i % 100 == 0:
                    print(f"Processing graph {i}/{dataset_size}")
                lsh_coor_array = lsh_features(coor_array[i], hash_size=hash_size, num_hash_tables=num_hash_tables)
                lsh_coor_tensor = torch.tensor(lsh_coor_array)
                num_nodes = lsh_coor_tensor.shape[0]
                k = min(int(0.5 * num_nodes), 99)
                edge_index = knn_graph(coor_tensor[i], k=k, loop=False)

                # Retrieve the feature vectors of the start and end nodes of each edge
                start_nodes = edge_index[0]
                end_nodes = edge_index[1]
                start_vectors = coor_tensor[i][start_nodes]
                end_vectors = coor_tensor[i][end_nodes]

                # Calculate RBF kernel values for each edge
                rbf_values = rbf_kernel(start_vectors, end_vectors, sigma=sigma)

                # Create a graph data object with RBF kernel values as edge attributes
                data = Data(x=lsh_coor_tensor, pos=coor_tensor[i], edge_index=edge_index, y=label_tensor[i], edge_attr=rbf_values)

                # Ensure the graph is undirected
                data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)

                graph_list.append(data)
            
    print("Creating final LSH dataset...")
    dataset = ClusterDataset(root=KNN_dataset_path, data_list=graph_list)
    print("LSH graph dataset generation complete!")
    return dataset

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Pooling at the graph level

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)

        return x

class GCN5(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN5, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 64)
        self.conv5 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Pooling at the graph level

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)

        return x

def run_KNN_test(datasets_folder_path, results_folder_path, coor_array, label_array, similarity_kernel, sigma=1):
    is_weighted = similarity_kernel != 'none'
    
    if is_weighted:
        KNN_dataset_name = f'adaptive_weighted_{similarity_kernel}'
    else:
        KNN_dataset_name = 'adaptive'
        
    if similarity_kernel == 'rbf':
        KNN_dataset_name += f'_sigma{sigma}'
        
    KNN_dataset_path = os.path.join(datasets_folder_path, KNN_dataset_name)
    if not os.path.exists(KNN_dataset_path):
        os.makedirs(KNN_dataset_path)

    KNN_dataset_file = os.path.join(KNN_dataset_path, 'processed')
    KNN_dataset_file = os.path.join(KNN_dataset_file, 'data.pt')

    if not os.path.exists(KNN_dataset_file):
        print(f'{KNN_dataset_name} graph dataset not found, generating...')
        dataset = generate_graph_dataset_lsh(coor_array, label_array, KNN_dataset_path, similarity_kernel, sigma, hash_size=6, num_hash_tables=8)
        print(f'{KNN_dataset_name} graph dataset generated and saved')
    else:
        print(f'{KNN_dataset_name} graph dataset found. Skip generating...')
        dataset = ClusterDataset(root=KNN_dataset_path)
        
    KNN_result_path = os.path.join(results_folder_path, KNN_dataset_name)

    if not os.path.exists(KNN_result_path):
        os.makedirs(KNN_result_path)

    train_proportion = 0.8
    trainset_size = int(dataset_size * train_proportion)
    testset_size = dataset_size - trainset_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [trainset_size, testset_size])
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    num_features = dataset[0].x.shape[1]
    num_classes = 2
    learning_rate = 0.00001
    model = GCN5(num_features, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    def train():
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            data.y = data.y.long()
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.y.size(0)

        return correct / total, total_loss

    def test(loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                data.y = data.y.long()
                out = model(data)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
                total += data.y.size(0)

        return correct / total
    
    def test_final(loader):
        model.eval()
        all_preds = []
        all_targets = []
        all_probabilities = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                data.y = data.y.long()
                out = model(data)
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(data.y.cpu().numpy())
                probs = torch.nn.functional.softmax(out, dim=1)[:, 1]
                all_probabilities.extend(probs.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probabilities)
        
        return accuracy, precision, recall, f1, auc
    
    losses = []
    train_accs = []
    test_accs = []
    num_epoches = 200

    print('Training start......')
    for epoch in range(num_epoches):
        train_acc, train_loss = train()
        test_acc = test(test_loader)
        print(f'Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    print('Training finished......')

    # Save trained model to local file
    model_save_path = os.path.join(KNN_result_path, 'model.pth')
    torch.save(model, model_save_path)
    print(f"trained model successfully saved to path {model_save_path}")

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # First subplot for losses
    axs[0].plot(losses, color='blue', label='Loss')
    axs[0].set_title('Losses Over Time')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)

    # Second subplot for train and test accuracies
    axs[1].plot(train_accs, color='green', label='Training Accuracy')
    axs[1].plot(test_accs, color='red', label='Test Accuracy')
    axs[1].set_title('Training and Test Accuracies')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()
    result_figure = os.path.join(KNN_result_path, 'figure.png')
    plt.savefig(result_figure)

    # Now call test function and print the metrics
    final_test_accuracy, final_precision, final_recall, final_f1_score, final_auc = test_final(test_loader)

    test_details = {
        'dataset_name': KNN_dataset_name,
        'dataset_path': KNN_dataset_path,
        'testset_size': testset_size,
        'number_of_features': num_features,
        'number_of_classes': num_classes,
        'number_of_epoches': num_epoches,
        'learning_rate': learning_rate,
        'metrics': {
            'Final Test Accuracy': final_test_accuracy,
            'Final Precision': final_precision,
            'Final Recall': final_recall,
            'Final F1 Score': final_f1_score,
            'Final AUC': final_auc
        }
    }

    # Print the test details with metrics
    print(json.dumps(test_details, indent=4))

    # Save the test details with metrics to a JSON file
    test_details_file = os.path.join(KNN_result_path, 'test_details.json')
    with open(test_details_file, 'w') as f:
        json.dump(test_details, f, indent=4)

if __name__ == "__main__":
    I_min = 5000  # min number of data points
    I_max = 5000  # max number of data points
    J = 50   # number of dimension
    max_K = 10   # max number of clusters
    dataset_size = 10000
    
    mean = 0.0
    std = 0.4

    cluster_type = 'k_means'
    
    raw_data_name = f'I{I_min}-{I_max}_J{J}_K{max_K}_{dataset_size}'
    
    root_path = os.getcwd()
    
    root_path = os.path.join(root_path, 'lsh_noisy')

    datasets_folder_path = os.path.join(root_path, 'datasets')
    datasets_folder_path = os.path.join(datasets_folder_path, cluster_type)
    datasets_folder_path = os.path.join(datasets_folder_path, raw_data_name)
    if not os.path.exists(datasets_folder_path):
        os.makedirs(datasets_folder_path)

    results_folder_path = os.path.join(root_path, 'results')
    results_folder_path = os.path.join(results_folder_path, cluster_type)
    results_folder_path = os.path.join(results_folder_path, raw_data_name)
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)
    
    raw_data_file = os.path.join(datasets_folder_path, 'data.npz')
    if not os.path.exists(raw_data_file):
        print(f'{cluster_type} synthetic data with specs {raw_data_name} not found, generating...')
        coor_array, label_array, I = generate_data.generate_data_gnn_balanced_varying_I(I_min, I_max, J, max_K, size=dataset_size, noise_type='gaussian', mean=mean, std=std) 
        np.savez(raw_data_file, coor_array=coor_array, label_array=label_array, I=I)
        print(f'{cluster_type} synthetic data with specs {raw_data_name} generated and saved as {raw_data_file}')
    else:
        print(f'{cluster_type} synthetic data with specs {raw_data_name} found in {raw_data_file}!')
        data = np.load(raw_data_file, allow_pickle=True)
        coor_array, label_array, I = data['coor_array'], data['label_array'], data['I']
    
    print(coor_array.shape)
    print(label_array.shape)
    print(I.shape)

    similarity_kernels = ['rbf']
    sigmas = [2]
    
    for similarity_kernel in similarity_kernels:
        if similarity_kernel == 'rbf':
                for sigma in sigmas:
                    run_KNN_test(datasets_folder_path, results_folder_path, coor_array, label_array, similarity_kernel, sigma)
        else:
            run_KNN_test(datasets_folder_path, results_folder_path, coor_array, label_array, similarity_kernel)
