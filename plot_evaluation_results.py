#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import os
from datetime import datetime

def load_results(json_file):
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def plot_from_json(json_file):
    """Create plots from JSON results file"""
    # Load data
    data = load_results(json_file)
    
    # Extract metadata and results
    metadata = data['metadata']
    results = data['results']
    
    # Get thresholds
    hopkins_thresholds = metadata['hopkins_thresholds']
    silhouette_thresholds = metadata['silhouette_thresholds']
    
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 14
    })
    
    # Create plots
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot Hopkins results
        hopkins_scores = [results['hopkins'][str(t)][metric][0] for t in hopkins_thresholds]
        ax.plot(hopkins_thresholds, hopkins_scores, 'b-', label='Hopkins', linewidth=2)
        
        # Plot Silhouette results
        silhouette_scores = [results['silhouette'][str(t)][metric][0] for t in silhouette_thresholds]
        ax.plot(silhouette_thresholds, silhouette_scores, 'g-', label='Silhouette', linewidth=2)
        
        # Plot GNN result as horizontal line
        gnn_score = results['gnn'][metric][0]
        ax.axhline(y=gnn_score, color='r', linestyle='--', label='GNN', linewidth=3)
        
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xlabel('Threshold')
        ax.set_ylabel(metric.capitalize())
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.dirname(json_file)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_name = os.path.join(output_dir, f'comparison_from_json_{timestamp}.png')
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {fig_name}")
    
    plt.show()

if __name__ == "__main__":
    json_file = "evaluation_noisy/I50_J2_K100_size1000/results_20241208_144258.json"
    # json_file = "/home/yluo147/projects/Starter-Project-Summer-2023/evaluation_noisy/I50_J30_K100_size1000/results_20241210_162709.json"

    plot_from_json(json_file)