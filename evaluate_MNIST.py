from evaluate import evaluate_hopkins, evaluate_kmeans_silhouette
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from hopkins_statistic import hopkins_statistic
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from main_gnn import generate_graph_dataset_lsh, ClusterDataset, GCN5
from torch_geometric.loader import DataLoader

def evaluate_gnn(model_path, data, device):
    """Modified version of evaluate_gnn with debugging info"""
    print("Evaluating GNN model...")
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    print("Converting data to graph dataset...")
    eval_folder = os.path.join('evaluation_mnist', 'graph_dataset')
    os.makedirs(eval_folder, exist_ok=True)
    
    # Delete existing dataset to force regeneration
    if os.path.exists(os.path.join(eval_folder, 'processed')):
        import shutil
        shutil.rmtree(os.path.join(eval_folder, 'processed'))
    
    print("Generating new graph dataset...")
    dataset = generate_graph_dataset_lsh(
        coor_array=data,
        label_array=np.zeros(len(data)),  # Dummy labels
        KNN_dataset_path=eval_folder,
        similarity_kernel='rbf',
        sigma=2,
        hash_size=6,
        num_hash_tables=8
    )
    
    # Print graph structure info
    data = dataset[0]
    print(f"\nGraph structure diagnostics:")
    print(f"Number of nodes: {data.x.shape[0]}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    print(f"Average degree: {data.edge_index.shape[1]/data.x.shape[0]:.2f}")
    print(f"Edge weights range: [{data.edge_attr.min():.4f}, {data.edge_attr.max():.4f}]")
    
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    print("\nRunning predictions...")
    all_preds = []
    all_probs = []  # Store prediction probabilities
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            probs = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Print prediction statistics
    print(f"\nPrediction statistics:")
    print(f"Average probability for class 0: {np.mean([p[0] for p in all_probs]):.4f}")
    print(f"Average probability for class 1: {np.mean([p[1] for p in all_probs]):.4f}")
    
    return all_preds



def load_and_preprocess_mnist():
    """Load MNIST data, flatten images, apply PCA and sample points"""
    print("Loading MNIST dataset...")
    mnist = datasets.MNIST(root='./data', train=True, download=True)
    X = mnist.data.numpy().reshape(-1, 28*28)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    
    # Random sampling
    print("Sampling 200 points...")
    np.random.seed(42)
    indices = np.random.choice(len(X_pca), size=200, replace=False)
    sampled_data = X_pca[indices]
    
    # Scale to [0, 10] range like synthetic data
    min_vals = np.min(sampled_data, axis=0)
    max_vals = np.max(sampled_data, axis=0)
    scaled_data = (sampled_data - min_vals) / (max_vals - min_vals) * 10
    
    # Print diagnostics
    print(f"MNIST scaled data range: [{np.min(scaled_data):.2f}, {np.max(scaled_data):.2f}]")
    print(f"MNIST scaled data mean: {np.mean(scaled_data):.2f}")
    print(f"MNIST scaled data std: {np.std(scaled_data):.2f}")
    
    return scaled_data

def generate_noise_data(reference_data):
    """Generate noise data in [0, 10] range"""
    noise_data = np.random.rand(*reference_data.shape) * 10
    
    # Print diagnostics
    print(f"Noise data range: [{np.min(noise_data):.2f}, {np.max(noise_data):.2f}]")
    print(f"Noise data mean: {np.mean(noise_data):.2f}")
    print(f"Noise data std: {np.std(noise_data):.2f}")
    
    return noise_data

def create_mixed_dataset(structured_data, noise_data, percentage, variant=1):
    """
    Create mixed dataset according to specified variant
    variant 1: 100% noise + p% structure
    variant 2: (100-p)% noise + p% structure
    """
    n_points = len(structured_data)
    n_structure = int(n_points * (percentage / 100))
    
    if variant == 1:
        # Take all noise points and add p% structure
        result = np.vstack([
            noise_data,
            structured_data[:n_structure]
        ])
    else:
        # Take (100-p)% noise and p% structure
        n_noise = n_points - n_structure
        result = np.vstack([
            noise_data[:n_noise],
            structured_data[:n_structure]
        ])
    
    # Print diagnostics for mixed dataset
    print(f"\nMixed dataset (variant {variant}, {percentage}% structure):")
    print(f"Shape: {result.shape}")
    print(f"Range: [{np.min(result):.2f}, {np.max(result):.2f}]")
    print(f"Mean: {np.mean(result):.2f}")
    print(f"Std: {np.std(result):.2f}")
    
    return result

def evaluate_mnist_structure(model_path, hopkins_thresholds, silhouette_thresholds, 
                           percentages=range(0, 101, 10), variants=[1, 2]):
    """Main evaluation function for MNIST structure detection"""
    print("\nStarting MNIST structure evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare data
    structured_data = load_and_preprocess_mnist()
    noise_data = generate_noise_data(structured_data)
    
    results = {
        variant: {
            'hopkins': {str(t): {str(p): {} for p in percentages} for t in hopkins_thresholds},
            'silhouette': {str(t): {str(p): {} for p in percentages} for t in silhouette_thresholds},
            'gnn': {str(p): {} for p in percentages}
        } for variant in variants
    }
    
    # Evaluate each variant
    for variant in variants:
        print(f"\nEvaluating variant {variant}...")
        
        for percentage in percentages:
            print(f"\nTesting with {percentage}% structure...")
            
            # Create mixed dataset
            current_data = create_mixed_dataset(structured_data, noise_data, percentage, variant)
            
            # Evaluate Hopkins
            for threshold in hopkins_thresholds:
                has_structure, score = evaluate_hopkins(current_data, threshold)
                results[variant]['hopkins'][str(threshold)][str(percentage)] = {
                    'prediction': int(has_structure),
                    'score': float(score)
                }
            
            # Evaluate Silhouette
            for threshold in silhouette_thresholds:
                has_structure, score, best_k = evaluate_kmeans_silhouette(current_data, threshold)
                results[variant]['silhouette'][str(threshold)][str(percentage)] = {
                    'prediction': int(has_structure),
                    'score': float(score),
                    'best_k': int(best_k)
                }
            
            # Evaluate GNN
            gnn_pred = evaluate_gnn(model_path, np.array([current_data]), device)
            results[variant]['gnn'][str(percentage)] = {
                'prediction': int(gnn_pred[0])
            }
    
    # Save results
    save_results(results, 'mnist_evaluation')
    plot_mnist_results_detailed(results, hopkins_thresholds, silhouette_thresholds, percentages, variants)
    
    return results

def save_results(results, prefix):
    """Save results to JSON file"""
    eval_folder = os.path.join('evaluation_mnist')
    os.makedirs(eval_folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(eval_folder, f'{prefix}_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_file}")

def plot_mnist_results_detailed(results, hopkins_thresholds, silhouette_thresholds, percentages, variants):
    """Plot both scores and binary predictions with larger fonts"""
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    for variant in variants:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))  # Made figure taller
        
        # Plot raw scores - only one line per method
        hopkins_scores = [results[variant]['hopkins'][str(hopkins_thresholds[0])][str(p)]['score'] 
                        for p in percentages]
        ax1.plot(percentages, hopkins_scores, '--', label='Hopkins Score', color='blue', linewidth=2)
        
        silhouette_scores = [results[variant]['silhouette'][str(silhouette_thresholds[0])][str(p)]['score'] 
                           for p in percentages]
        ax1.plot(percentages, silhouette_scores, ':', label='Silhouette Score', color='green', linewidth=2)
        
        # Add horizontal lines for thresholds
        for threshold in hopkins_thresholds:
            ax1.axhline(y=threshold, color='blue', linestyle=':', alpha=0.3)
            ax1.text(5, threshold, f'H-t={threshold:.2f}', color='blue', alpha=0.7, fontsize=12)
            
        for threshold in silhouette_thresholds:
            ax1.axhline(y=threshold, color='green', linestyle=':', alpha=0.3)
            ax1.text(5, threshold, f'S-t={threshold:.2f}', color='green', alpha=0.7, fontsize=12)
        
        ax1.set_title(f'Variant {variant}: Raw Scores with Thresholds', pad=20)
        ax1.set_xlabel('Percentage of Structured Data', labelpad=10)
        ax1.set_ylabel('Score', labelpad=10)
        ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        
        # Plot binary predictions
        for threshold in hopkins_thresholds:
            hopkins_preds = [results[variant]['hopkins'][str(threshold)][str(p)]['prediction'] 
                           for p in percentages]
            ax2.plot(percentages, hopkins_preds, '--', label=f'Hopkins (t={threshold:.2f})', linewidth=2)
        
        for threshold in silhouette_thresholds:
            silhouette_preds = [results[variant]['silhouette'][str(threshold)][str(p)]['prediction'] 
                              for p in percentages]
            ax2.plot(percentages, silhouette_preds, ':', label=f'Silhouette (t={threshold:.2f})', linewidth=2)
        
        gnn_preds = [results[variant]['gnn'][str(p)]['prediction'] for p in percentages]
        ax2.plot(percentages, gnn_preds, 'r-', label='GNN', linewidth=3)
        
        ax2.set_title(f'Variant {variant}: Binary Predictions', pad=20)
        ax2.set_xlabel('Percentage of Structured Data', labelpad=10)
        ax2.set_ylabel('Prediction', labelpad=10)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['No Structure', 'Structure'], fontsize=12)
        ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)  # Added more padding between subplots
        
        # Save plot
        eval_folder = os.path.join('evaluation_mnist')
        os.makedirs(eval_folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_name = os.path.join(eval_folder, f'mnist_comparison_variant{variant}_{timestamp}.png')
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    print("Starting MNIST evaluation script...")
    
    # Configuration
    model_path = "/home/yluo147/projects/Starter-Project-Summer-2023/lsh_noisy/results/k_means/I50-200_J50_K100_100000/adaptive_weighted_rbf_sigma2/model.pth"  # Update with your model path
    hopkins_thresholds = np.arange(0.6, 0.9, 0.05)
    silhouette_thresholds = np.arange(0.3, 0.8, 0.05)
    percentages = range(0, 101, 10)  # 0% to 100% in steps of 10%
    variants = [1, 2]  # Test both variants
    
    # Run evaluation
    results = evaluate_mnist_structure(
        model_path,
        hopkins_thresholds,
        silhouette_thresholds,
        percentages,
        variants
    )