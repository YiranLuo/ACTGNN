import os
import json
import glob

def get_latest_results_file(folder_path):
    """Get the most recent results json file from the given folder."""
    json_files = glob.glob(os.path.join(folder_path, "results_*.json"))
    if not json_files:
        return None
    
    latest_file = max(json_files, key=lambda x: os.path.basename(x).split('_')[1:3])
    return latest_file

def get_best_metrics(results_dict):
    """Get all metrics for the threshold that produced the best f1 score."""
    best_f1 = -1
    best_threshold = None
    best_metrics = None
    
    for threshold, metrics in results_dict.items():
        f1_score = metrics['f1'][0]
        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold
            best_metrics = {
                'accuracy': metrics['accuracy'][0],
                'precision': metrics['precision'][0],
                'recall': metrics['recall'][0],
                'f1': f1_score
            }
    
    return float(best_threshold), best_metrics

def summarize_results():
    base_dir = "evaluation_noisy"
    i50_folders = glob.glob(os.path.join(base_dir, "I50*"))
    
    results_summary = []
    
    for folder in i50_folders:
        folder_name = os.path.basename(folder)
        latest_file = get_latest_results_file(folder)
        
        if not latest_file:
            continue
            
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        metadata = data['metadata']
        results = data['results']
        
        # Get best metrics for each method
        hopkins_threshold, hopkins_metrics = get_best_metrics(results['hopkins'])
        silhouette_threshold, silhouette_metrics = get_best_metrics(results['silhouette'])
        gnn_metrics = {
            'accuracy': results['gnn']['accuracy'][0],
            'precision': results['gnn']['precision'][0],
            'recall': results['gnn']['recall'][0],
            'f1': results['gnn']['f1'][0]
        }
        
        results_summary.append({
            'folder': folder_name,
            'I': metadata['I'],
            'J': metadata['J'],
            'K': metadata['max_K'],
            'size': metadata['size'],
            'hopkins': {'threshold': hopkins_threshold, 'metrics': hopkins_metrics},
            'silhouette': {'threshold': silhouette_threshold, 'metrics': silhouette_metrics},
            'gnn': {'metrics': gnn_metrics}
        })
    
    # Print summary
    print("\nEvaluation Results Summary for I50 Experiments")
    print("=" * 80)
    
    for result in results_summary:
        print(f"\nFolder: {result['folder']}")
        print(f"Parameters: I={result['I']}, J={result['J']}, K={result['K']}, size={result['size']}")
        print("\nHopkins Statistics (threshold={:.2f}):".format(result['hopkins']['threshold']))
        print(f"  Accuracy : {result['hopkins']['metrics']['accuracy']:.4f}")
        print(f"  Precision: {result['hopkins']['metrics']['precision']:.4f}")
        print(f"  Recall   : {result['hopkins']['metrics']['recall']:.4f}")
        print(f"  F1 Score : {result['hopkins']['metrics']['f1']:.4f}")
        
        print("\nSilhouette Score (threshold={:.2f}):".format(result['silhouette']['threshold']))
        print(f"  Accuracy : {result['silhouette']['metrics']['accuracy']:.4f}")
        print(f"  Precision: {result['silhouette']['metrics']['precision']:.4f}")
        print(f"  Recall   : {result['silhouette']['metrics']['recall']:.4f}")
        print(f"  F1 Score : {result['silhouette']['metrics']['f1']:.4f}")
        
        print("\nGNN Results:")
        print(f"  Accuracy : {result['gnn']['metrics']['accuracy']:.4f}")
        print(f"  Precision: {result['gnn']['metrics']['precision']:.4f}")
        print(f"  Recall   : {result['gnn']['metrics']['recall']:.4f}")
        print(f"  F1 Score : {result['gnn']['metrics']['f1']:.4f}")
        print("-" * 80)

if __name__ == "__main__":
    summarize_results()