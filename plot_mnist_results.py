import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_mnist_results_from_file(json_path):
    """Plot results from saved JSON file with enhanced visualization"""
    # Load results from JSON file
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Extract parameters
    variants = sorted([int(k) for k in results.keys()])
    hopkins_thresholds = sorted([float(t) for t in results['1']['hopkins'].keys()])
    silhouette_thresholds = sorted([float(t) for t in results['1']['silhouette'].keys()])
    percentages = sorted([int(p) for p in results['1']['hopkins'][str(hopkins_thresholds[0])].keys()])
    
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 30,
        'axes.titlesize': 40,
        'axes.labelsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 14
    })
    
    for variant in variants:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
        
        # Plot raw scores
        hopkins_scores = [results[str(variant)]['hopkins'][str(hopkins_thresholds[0])][str(p)]['score'] 
                        for p in percentages]
        ax1.plot(percentages, hopkins_scores, '--', label='Hopkins Score', color='blue', linewidth=2)
        
        silhouette_scores = [results[str(variant)]['silhouette'][str(silhouette_thresholds[0])][str(p)]['score'] 
                           for p in percentages]
        ax1.plot(percentages, silhouette_scores, ':', label='Silhouette Score', color='green', linewidth=2)
        
        # Add horizontal lines for thresholds
        y_min, y_max = ax1.get_ylim()
        
        # Hopkins thresholds on the left side
        for idx, threshold in enumerate(hopkins_thresholds):
            ax1.axhline(y=threshold, color='blue', linestyle=':', alpha=0.3, xmax=0.02)
            ax1.text(5, threshold, f'H-t={threshold:.2f}', color='blue', 
                    alpha=0.7, fontsize=14, ha='left', va='center')
            
        # Silhouette thresholds on the right side
        for idx, threshold in enumerate(silhouette_thresholds):
            ax1.axhline(y=threshold, color='green', linestyle=':', alpha=0.3, xmin=0.98)
            ax1.text(95, threshold, f'S-t={threshold:.2f}', color='green', 
                    alpha=0.7, fontsize=14, ha='right', va='center')
        
        ax1.set_title(f'Variant {variant}: Raw Scores with Thresholds', pad=20)
        ax1.set_xlabel('Percentage of Structured Data', labelpad=15)
        ax1.set_ylabel('Score', labelpad=15)
        ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)
        
        # Plot binary predictions with legend outside
        for threshold in hopkins_thresholds:
            hopkins_preds = [results[str(variant)]['hopkins'][str(threshold)][str(p)]['prediction'] 
                           for p in percentages]
            ax2.plot(percentages, hopkins_preds, '--', label=f'Hopkins (t={threshold:.2f})', linewidth=2)
        
        for threshold in silhouette_thresholds:
            silhouette_preds = [results[str(variant)]['silhouette'][str(threshold)][str(p)]['prediction'] 
                              for p in percentages]
            ax2.plot(percentages, silhouette_preds, ':', label=f'Silhouette (t={threshold:.2f})', linewidth=2)
        
        gnn_preds = [results[str(variant)]['gnn'][str(p)]['prediction'] for p in percentages]
        ax2.plot(percentages, gnn_preds, 'r-', label='GNN', linewidth=3)
        
        ax2.set_title(f'Variant {variant}: Binary Predictions', pad=20)
        ax2.set_xlabel('Percentage of Structured Data', labelpad=15)
        ax2.set_ylabel('Prediction', labelpad=15)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['No Structure', 'Structure'], fontsize=16)
        # Move legend outside of the plot
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Adjust layout to prevent legend cutoff
        plt.subplots_adjust(right=0.85)
        
        # Save plot
        output_dir = os.path.dirname(json_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_name = os.path.join(output_dir, f'mnist_comparison_variant{variant}_{timestamp}.png')
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    json_path = "/home/yluo147/projects/Starter-Project-Summer-2023/evaluation_mnist/mnist_evaluation_results_20241213_164712.json"
    plot_mnist_results_from_file(json_path) 