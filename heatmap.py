#!/usr/bin/env python3

import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def extract_knn_percentage(folder_name):
    """Extract KNN percentage from folder name"""
    return int(folder_name.split('KNN')[1].split('_')[0])

def get_edge_feature_type(folder_name):
    """Determine edge feature type from folder name"""
    if '_weighted_rbf_sigma' in folder_name:
        sigma = folder_name.split('sigma')[1]
        return f'RBF σ={sigma}'
    elif '_weighted_cosine' in folder_name:
        return 'Cosine'
    elif '_weighted' in folder_name:
        return 'Euclidean'
    else:
        return 'Unweighted'

def load_accuracy(file_path):
    """Load accuracy from test_details.json file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data['metrics']['Final Test Accuracy']
    except:
        return None

def create_heatmap():
    # Define base directory
    base_dir = "/home/yluo147/projects/Starter-Project-Summer-2023/results/k_means/I100_J2_K3_20000"
    
    # Define axes values
    knn_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    edge_features = ['Unweighted', 'Euclidean', 'Cosine', 
                    'RBF σ=0.5', 'RBF σ=1', 'RBF σ=2', 'RBF σ=5']
    
    # Create empty matrix for accuracies
    accuracies = np.zeros((len(edge_features), len(knn_percentages)))
    accuracies.fill(np.nan)  # Fill with NaN to distinguish missing data
    
    # Scan directories and fill matrix
    for folder in os.listdir(base_dir):
        if folder.startswith('KNN'):
            # Get KNN percentage
            knn_pct = extract_knn_percentage(folder)
            if knn_pct not in knn_percentages:
                continue
                
            # Get edge feature type
            edge_type = get_edge_feature_type(folder)
            if edge_type not in edge_features:
                continue
            
            # Load accuracy
            json_path = os.path.join(base_dir, folder, 'test_details.json')
            accuracy = load_accuracy(json_path)
            
            if accuracy is not None:
                i = edge_features.index(edge_type)
                j = knn_percentages.index(knn_pct)
                accuracies[i, j] = accuracy * 100  # Convert to percentage
    
    # Create heatmap
    plt.figure(figsize=(15, 10))
    
    # Set font sizes
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18
    })
    
    # Create heatmap
    sns.heatmap(accuracies, 
                annot=True, 
                fmt='.1f',
                cmap='YlOrRd',
                xticklabels=knn_percentages,
                yticklabels=edge_features,
                cbar_kws={'label': 'Accuracy (%)'},
                mask=np.isnan(accuracies))  # Mask missing values
    
    plt.xlabel('Percentage of Nearest Neighbors Connected (%)')
    plt.ylabel('Edge Feature Type')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(base_dir, f'accuracy_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    create_heatmap()