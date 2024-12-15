import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from hopkins_statistic import hopkins_statistic
import generate_data
from main_gnn import generate_graph_dataset_lsh, ClusterDataset, GCN5
from torch_geometric.loader import DataLoader
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
from threadpoolctl import threadpool_limits
import json
from datetime import datetime

import warnings
import os
from threadpoolctl import threadpool_limits
import logging


def evaluate_hopkins(data, threshold):
    """
    Evaluate clustering tendency using Hopkins statistic
    Returns True if clustering tendency is detected, False otherwise
    """
    h_score = hopkins_statistic(data)
    return h_score > threshold, h_score

def evaluate_kmeans_silhouette(data, threshold, k_range=None):
    """
    Evaluate clustering tendency using K-means + silhouette score
    Returns True if clustering tendency is detected, False otherwise
    """
    # If k_range not provided, use sqrt rule of thumb with a reasonable upper limit
    if k_range is None:
        n_samples = len(data)
        min_k = 2
        # Rule of thumb: sqrt of n/2 as upper limit, capped at 20 for computational efficiency
        max_k = min(20, int(np.sqrt(n_samples/2)))
        k_range = range(min_k, max_k + 1)
    
    max_score = -1
    best_k = -1
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        
        if score > max_score:
            max_score = score
            best_k = k
            
    return max_score > threshold, max_score, best_k

def evaluate_gnn(model_path, data, device, I, J, max_K, size, KNN):
    """
    Evaluate clustering tendency using the trained GNN model
    """
    print("Evaluating GNN model...")
    print(f"Loading model from {model_path}")
    # Load the trained model
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    print("Converting data to graph dataset...")
    # Create graph dataset path under evaluation folder
    eval_folder = os.path.join('evaluation_noisy', f'I{I}_J{J}_K{max_K}_size{size}')
    os.makedirs(eval_folder, exist_ok=True)
    graph_dataset_path = os.path.join(eval_folder, 'graph_dataset')
    
    # Check if dataset already exists
    dataset_file = os.path.join(graph_dataset_path, 'processed', 'data.pt')
    if os.path.exists(dataset_file):
        print("Graph dataset found. Skip generating...")
        dataset = ClusterDataset(root=graph_dataset_path)
    else:
        print("Graph dataset not found, generating...")
        dataset = generate_graph_dataset_lsh(
            coor_array=data,
            label_array=np.zeros(len(data)),  # Dummy labels
            KNN_dataset_path=graph_dataset_path,
            similarity_kernel='rbf',
            sigma=2,
            hash_size=6,
            num_hash_tables=8
        )
        print("Graph dataset generated and saved")
    
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    print("Running predictions...")
    all_preds = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            
    return all_preds

def generate_test_datasets(I, J, max_K, size, mean, std):
    """
    Generate balanced test datasets using generate_data_gnn_balanced
    Returns datasets and their labels (1 for clustered, 0 for non-clustered)
    """
    print(f"Generating {size} test datasets...")
    
    # Create folder structure
    eval_folder = os.path.join('evaluation_noisy', f'I{I}_J{J}_K{max_K}_size{size}')
    os.makedirs(eval_folder, exist_ok=True)
    
    raw_data_file = os.path.join(eval_folder, 'data.npz')
    
    if not os.path.exists(raw_data_file):
        print(f'Synthetic test data with specs I{I}_J{J}_K{max_K}_size{size} not found, generating...')
        data, labels = generate_data.generate_data_gnn_balanced(I, J, max_K, size=size, noise_type='gaussian', mean=mean, std=std)
        np.savez(raw_data_file, coor_array=data, label_array=labels)
        print(f'Synthetic test data generated and saved as {raw_data_file}')
    else:
        print(f'Synthetic test data found in {raw_data_file}!')
        loaded_data = np.load(raw_data_file)
        data, labels = loaded_data['coor_array'], loaded_data['label_array']
    
    # Convert to list of individual datasets
    datasets = [data[i] for i in range(len(data))]
    print("Test datasets loaded successfully")
    
    return datasets, labels

def compare_methods(model_path, hopkins_thresholds, silhouette_thresholds, I, J, max_K, size, mean, std, KNN):
    """
    Compare all methods on the test datasets
    """
    print("\nStarting method comparison...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    # Generate test datasets
    datasets, true_labels = generate_test_datasets(
        I=I,
        J=J,
        max_K=max_K,
        size=size,
        mean=mean,
        std=std
    )
    
    results = {
        'hopkins': {str(t): {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for t in hopkins_thresholds},
        'silhouette': {str(t): {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for t in silhouette_thresholds},
        'gnn': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }
    
    # Evaluate each method
    print("\nEvaluating Hopkins method...")
    for threshold in hopkins_thresholds:
        print(f"\nTesting Hopkins threshold: {threshold}")
        hopkins_preds = []
        total_score = 0
        for i, data in enumerate(datasets):
            pred, score = evaluate_hopkins(data, threshold)
            hopkins_preds.append(1 if pred else 0)
            total_score += score
            
            if (i + 1) % (len(datasets) // 4) == 0:  # Print progress at 25%, 50%, 75%, 100%
                avg_score = total_score / (i + 1)
                print(f"Progress: {(i+1)/len(datasets)*100:.1f}% | Average Hopkins score: {avg_score:.3f}")
        
        results['hopkins'][str(threshold)]['accuracy'].append(accuracy_score(true_labels, hopkins_preds))
        results['hopkins'][str(threshold)]['precision'].append(precision_score(true_labels, hopkins_preds))
        results['hopkins'][str(threshold)]['recall'].append(recall_score(true_labels, hopkins_preds))
        results['hopkins'][str(threshold)]['f1'].append(f1_score(true_labels, hopkins_preds))
    
    print("\nEvaluating Silhouette method...")
    for threshold in silhouette_thresholds:
        print(f"\nTesting Silhouette threshold: {threshold}")
        silhouette_preds = []
        total_score = 0
        total_k = 0
        for i, data in enumerate(datasets):
            pred, score, best_k = evaluate_kmeans_silhouette(data, threshold)
            silhouette_preds.append(1 if pred else 0)
            total_score += score
            total_k += best_k
            
            if (i + 1) % (len(datasets) // 4) == 0:  # Print progress at 25%, 50%, 75%, 100%
                avg_score = total_score / (i + 1)
                avg_k = total_k / (i + 1)
                print(f"Progress: {(i+1)/len(datasets)*100:.1f}% | Average Silhouette score: {avg_score:.3f} | Average optimal k: {avg_k:.1f}")
        
        results['silhouette'][str(threshold)]['accuracy'].append(accuracy_score(true_labels, silhouette_preds))
        results['silhouette'][str(threshold)]['precision'].append(precision_score(true_labels, silhouette_preds))
        results['silhouette'][str(threshold)]['recall'].append(recall_score(true_labels, silhouette_preds))
        results['silhouette'][str(threshold)]['f1'].append(f1_score(true_labels, silhouette_preds))
    
    print("\nEvaluating GNN method...")
    gnn_preds = evaluate_gnn(model_path, np.array(datasets), device, I, J, max_K, size, KNN)
    
    results['gnn']['accuracy'].append(accuracy_score(true_labels, gnn_preds))
    results['gnn']['precision'].append(precision_score(true_labels, gnn_preds))
    results['gnn']['recall'].append(recall_score(true_labels, gnn_preds))
    results['gnn']['f1'].append(f1_score(true_labels, gnn_preds))
    
    # Save results to JSON file
    eval_folder = os.path.join('evaluation_noisy', f'I{I}_J{J}_K{max_K}_size{size}')
    os.makedirs(eval_folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(eval_folder, f'results_{timestamp}.json')
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = {
        'metadata': {
            'I': I,
            'J': J,
            'max_K': max_K,
            'size': size,
            'model_path': model_path,
            'hopkins_thresholds': hopkins_thresholds.tolist(),
            'silhouette_thresholds': silhouette_thresholds.tolist()
        },
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=4)
    print(f"\nResults saved to {results_file}")
    
    print("\nComparison completed successfully")
    return results

def plot_results(results, hopkins_thresholds, silhouette_thresholds, I, J, max_K, size):
    """
    Plot comparison results
    """
    print("\nGenerating comparison plots...")
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        print(f"Plotting {metric} comparison...")
        ax = axes[idx]
        
        # Plot Hopkins results
        hopkins_scores = [results['hopkins'][str(t)][metric][0] for t in hopkins_thresholds]
        ax.plot(hopkins_thresholds, hopkins_scores, 'b-', label='Hopkins')
        
        # Plot Silhouette results
        silhouette_scores = [results['silhouette'][str(t)][metric][0] for t in silhouette_thresholds]
        ax.plot(silhouette_thresholds, silhouette_scores, 'g-', label='Silhouette')
        
        # Plot GNN result as horizontal line
        gnn_score = results['gnn'][metric][0]
        ax.axhline(y=gnn_score, color='r', linestyle='--', label='GNN')
        
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xlabel('Threshold')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    eval_folder = os.path.join('evaluation_noisy', f'I{I}_J{J}_K{max_K}_size{size}')
    os.makedirs(eval_folder, exist_ok=True)
    
    # Save figure with descriptive name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_name = os.path.join(eval_folder, f'comparison_size{size}_{timestamp}.png')
    plt.savefig(fig_name)
    print(f"Figure saved as {fig_name}")
    
    print("Displaying plots...")
    plt.show()

# Usage example:
if __name__ == "__main__":
    print("Starting evaluation script...")
    # model_path = "/data/home/yluo147/projects/Starter-Project-Summer-2023/lsh_noisy/results/k_means/I200_J2_K100_20000/KNN99_weighted_rbf_sigma2/model.pth"
    # model_path = "/home/yluo147/projects/Starter-Project-Summer-2023/lsh_noisy/results/k_means/I50-200_J2_K100_100000/adaptive_weighted_rbf_sigma2/model.pth"
    model_path = "/home/yluo147/projects/Starter-Project-Summer-2023/lsh_noisy/results/k_means/I50-200_J50_K100_100000/adaptive_weighted_rbf_sigma2/model.pth"
    hopkins_thresholds = np.arange(0.6, 0.9, 0.05)
    silhouette_thresholds = np.arange(0.3, 0.8, 0.05)
    
    # Test dataset parameters
    I = 300
    J = 50
    max_K = 100
    size = 1000
    
    mean = 0.0
    std = 0.4
    KNN = min(int(0.5*I), 99)
    
    results = compare_methods(model_path, hopkins_thresholds, silhouette_thresholds, I, J, max_K, size, mean, std, KNN)
    plot_results(results, hopkins_thresholds, silhouette_thresholds, I, J, max_K, size)
