import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import seaborn as sns
from glob import glob
from tqdm import tqdm
from cheatools.lgnn import lGNN

def prepare_graph_for_model(graph, target_feature_dim=None):
   
    graph_copy = graph.clone()
    
    if target_feature_dim is None:
        if hasattr(graph_copy, 'energy'):
            graph_copy.y = graph_copy.energy
        return graph_copy
    
    current_dim = graph_copy.x.shape[1]
    
    if current_dim < target_feature_dim:
        padding = torch.zeros((graph_copy.x.shape[0], target_feature_dim - current_dim), 
                             dtype=graph_copy.x.dtype)
        graph_copy.x = torch.cat([graph_copy.x, padding], dim=1)
    elif current_dim > target_feature_dim:
        graph_copy.x = graph_copy.x[:, :target_feature_dim]
        print(f"Warning: Truncating node features from {current_dim} to {target_feature_dim} dimensions")
    
    if hasattr(graph_copy, 'energy'):
        graph_copy.y = graph_copy.energy
    
    return graph_copy

def load_site_specific_graphs(graph_dir, adsorbate_type='H', model_feature_dim=None):
   
    all_graphs = []
    filenames = []
    
    pattern = os.path.join(graph_dir, f"*_{adsorbate_type}_site*.pt")
    for graph_file in tqdm(glob(pattern), desc=f"Loading {adsorbate_type} graphs from {os.path.basename(graph_dir)}"):
        try:
            graph = torch.load(graph_file)
            
            if hasattr(graph, 'energy'):
                if model_feature_dim is not None:
                    prepared_graph = prepare_graph_for_model(graph, model_feature_dim)
                else:
                    prepared_graph = graph
                    if hasattr(graph, 'energy'):
                        prepared_graph.y = graph.energy
                
                all_graphs.append(prepared_graph)
                filenames.append(os.path.basename(graph_file))
            else:
                print(f"Warning: {graph_file} has no energy data. Skipping.")
        except Exception as e:
            print(f"Error loading {graph_file}: {str(e)}")
    
    return all_graphs, filenames

def evaluate_metrics(predictions, targets):

    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    r2 = r2_score(targets, predictions)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Mean Error': np.mean(predictions - targets),
        'Max Error': np.max(np.abs(predictions - targets)),
        'Min Error': np.min(np.abs(predictions - targets))
    }

def create_parity_plot(predictions, targets, category, adsorbate_type, output_dir):
  
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(targets, predictions, alpha=0.7, edgecolor='black')
    
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    buffer = (max_val - min_val) * 0.05
    ax.plot([min_val-buffer, max_val+buffer], [min_val-buffer, max_val+buffer], 'r--')
    
    ax.set_xlabel('DFT Energy (eV)', fontsize=14)
    ax.set_ylabel('Predicted Energy (eV)', fontsize=14)
    ax.set_title(f'{adsorbate_type} Adsorption Energy on {category.capitalize()}', fontsize=16)
    
    metrics = evaluate_metrics(predictions, targets)
    metrics_text = f"MAE: {metrics['MAE']:.3f} eV\nRMSE: {metrics['RMSE']:.3f} eV\nR²: {metrics['R²']:.3f}"
    ax.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                va='top', ha='left', fontsize=12)
    
    ax.axis('equal')
    ax.axis('square')
    
    os.makedirs(os.path.join(output_dir, 'test_results'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'test_results', f'{adsorbate_type}_on_{category}_parity.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_detailed_results(predictions, targets, filenames, category, adsorbate_type, output_dir):
 
    results_df = pd.DataFrame({
        'Filename': filenames,
        'DFT_Energy': targets,
        'Predicted_Energy': predictions,
        'Error': np.array(predictions) - np.array(targets),
        'Abs_Error': np.abs(np.array(predictions) - np.array(targets))
    })
    
    results_df = results_df.sort_values('Abs_Error', ascending=False)
    
    os.makedirs(os.path.join(output_dir, 'test_results'), exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'test_results', f'{adsorbate_type}_on_{category}_detailed_results.csv'), index=False)
    
    return results_df

def test_model_on_category(model, graphs, batch_size=64):
   
    if len(graphs) == 0:
        return [], [], []
        
    batch_size = min(batch_size, len(graphs))
    
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    predictions, targets, ids = model.test(loader, batch_size)
    return predictions, targets, ids

def train_model_on_site_specific_data(
    data_dirs, adsorbate_type, output_dir='models', 
    batch_size=32, learning_rate=0.001, epochs=100,
    train_ratio=0.8, val_ratio=0.1
):
    """Train a model on site-specific adsorption data"""
    
    all_graphs = []
    all_filenames = []
    
    for category, data_dir in data_dirs.items():
        graphs, filenames = load_site_specific_graphs(data_dir, adsorbate_type)
        all_graphs.extend(graphs)
        all_filenames.extend([f"{category}/{f}" for f in filenames])
    
    print(f"Loaded total of {len(all_graphs)} {adsorbate_type} adsorption graphs")
    
    if len(all_graphs) == 0:
        print("No graphs to train on!")
        return
    
    feature_dim = all_graphs[0].x.shape[1]
    
    arch = {
        'n_conv_layers': 3,
        'n_hidden_layers': 1,
        'conv_dim': 64,
        'act': 'relu',
        'harmonic': True,
        'input_dim': feature_dim
    }
    
    indices = torch.randperm(len(all_graphs))
    train_size = int(len(all_graphs) * train_ratio)
    val_size = int(len(all_graphs) * val_ratio)
    test_size = len(all_graphs) - train_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_graphs = [all_graphs[i] for i in train_indices]
    val_graphs = [all_graphs[i] for i in val_indices]
    test_graphs = [all_graphs[i] for i in test_indices]
    test_filenames = [all_filenames[i] for i in test_indices]
    
    print(f"Split: Train: {len(train_graphs)}, Validation: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)
    
    model = lGNN(arch=arch)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {adsorbate_type} adsorption model")
    for epoch in range(epochs):
        train_loss = model.train4epoch(train_loader, min(batch_size, len(train_graphs)), optimizer)
        train_losses.append(train_loss)
        
        model.eval()
        val_pred, val_target, _ = model.test(val_loader, min(batch_size, len(val_graphs)))
        val_loss = mean_absolute_error(val_target, val_pred)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            model_state = {
                'arch': arch,
                'onehot_labels': all_graphs[0].onehot_labels,
                'adsorbate_type': adsorbate_type
            }
            model_state.update(model.state_dict())
            
            model_path = os.path.join(output_dir, f'lGNN_{adsorbate_type}_adsorption.state')
            torch.save(model_state, model_path)
    
    print(f"\nTraining complete. Best validation MAE: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    model_state = torch.load(os.path.join(output_dir, f'lGNN_{adsorbate_type}_adsorption.state'))
    model_state_clean = {k: v for k, v in model_state.items() 
                        if k not in ['onehot_labels', 'arch', 'adsorbate_type']}
    
    model = lGNN(arch=model_state['arch'])
    model.load_state_dict(model_state_clean, strict=False)
    model.eval()
    
    test_pred, test_target, _ = model.test(test_loader, min(batch_size, len(test_graphs)))
    test_metrics = evaluate_metrics(test_pred, test_target)
    
    print(f"\nTest Results for {adsorbate_type} adsorption:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    os.makedirs(os.path.join(output_dir, 'training_results'), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train MAE')
    plt.plot(val_losses, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (eV)')
    plt.title(f'Learning Curve for {adsorbate_type} Adsorption')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'training_results', f'{adsorbate_type}_learning_curve.png'), 
                dpi=300, bbox_inches='tight')
    
    create_parity_plot(test_pred, test_target, 'test', adsorbate_type, output_dir)
    
    save_detailed_results(test_pred, test_target, test_filenames, 'test', adsorbate_type, output_dir)
    
    return model, test_metrics

def test_model_on_site_specific_data(model_path, data_dirs, adsorbate_type, output_dir='models'):
 
    print(f"Loading model from {model_path}")
    model_state = torch.load(model_path)
    
    arch = model_state.get('arch', {
        'n_conv_layers': 3,
        'n_hidden_layers': 0,
        'conv_dim': 18,
        'act': 'relu',
        'harmonic': True,
        'input_dim': 13
    })
    
    model = lGNN(arch=arch)
    
    model_state_clean = {k: v for k, v in model_state.items() 
                        if k not in ['onehot_labels', 'arch', 'adsorbate_type']}
    
    model.load_state_dict(model_state_clean, strict=False)
    model.eval()  
    
    summary_results = []
    all_df_results = []
    
    for category, data_dir in data_dirs.items():
        print(f"\nTesting on {category} data...")
        
        test_graphs, filenames = load_site_specific_graphs(data_dir, adsorbate_type)
        
        if len(test_graphs) > 0:
            print(f"Loaded {len(test_graphs)} graphs for testing")
            
            predictions, targets, _ = test_model_on_category(model, test_graphs)
            
            metrics = evaluate_metrics(predictions, targets)
            metrics['Category'] = category
            metrics['Num Samples'] = len(test_graphs)
            summary_results.append(metrics)
            
            print(f"Results for {category}:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    print(f"  {metric_name}: {metric_value:.4f}")
                else:
                    print(f"  {metric_name}: {metric_value}")
            
            create_parity_plot(predictions, targets, category, adsorbate_type, output_dir)
            
            df_results = save_detailed_results(predictions, targets, filenames, category, adsorbate_type, output_dir)
            all_df_results.append(df_results)
        else:
            print(f"No valid graphs found for {category}")
    
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(os.path.join(output_dir, 'test_results', f'{adsorbate_type}_summary_results.csv'), index=False)
        
        if len(summary_results) > 1:
            create_category_comparison_plot(summary_df, adsorbate_type, output_dir)
    
    return summary_results

def train_with_configuration(
    train_config="all", 
    test_configs=None,
    adsorbate_type="H", 
    base_data_dirs=None,
    output_dir='models', 
    batch_size=32, 
    learning_rate=0.0005, 
    epochs=200,
    skip_training=False
):
   
    if base_data_dirs is None:
        base_data_dirs = {
        'pairs': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/pairs',
        'triplets': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/triplets',
        'fives': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/fives'
        }
    
    if test_configs is None:
        if train_config == "all":
            test_configs = ["pairs", "triplets", "fives"]
        else:
            test_configs = [train_config]
    elif test_configs == "all":
        test_configs = ["pairs", "triplets", "fives"]
    elif isinstance(test_configs, str):
        test_configs = [test_configs]
    
    model_subdir = os.path.join(output_dir, f"{train_config}_{adsorbate_type}")
    os.makedirs(model_subdir, exist_ok=True)
    model_path = os.path.join(model_subdir, f'lGNN_{adsorbate_type}_adsorption.state')
    
    if train_config == "all":
        train_data_dirs = base_data_dirs
    elif train_config in ["pairs", "triplets", "fives"]:
        train_data_dirs = {train_config: base_data_dirs[train_config]}
    else:
        raise ValueError(f"Invalid training configuration: {train_config}")
    
    if not skip_training:
        print(f"\n=== Training {adsorbate_type} adsorption model on {train_config} configuration ===")
        model, metrics = train_model_on_site_specific_data(
            train_data_dirs, 
            adsorbate_type, 
            model_subdir, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            epochs=epochs
        )
    else:
        print(f"\n=== Using existing {adsorbate_type} model trained on {train_config} configuration ===")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Cannot skip training.")
    
    print(f"\n=== Testing {adsorbate_type} model on {', '.join(test_configs)} ===")
    all_results = {}
    
    for test_config in test_configs:
        if test_config not in base_data_dirs:
            print(f"Warning: Test configuration '{test_config}' not found in data directories. Skipping.")
            continue
            
        test_data_dirs = {test_config: base_data_dirs[test_config]}
        results = test_model_on_site_specific_data(
            model_path, 
            test_data_dirs, 
            adsorbate_type, 
            os.path.join(model_subdir, f"test_on_{test_config}")
        )
        all_results[test_config] = results
    
    return model_path, all_results

def create_category_comparison_plot(summary_df, adsorbate_type, output_dir):
   
    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = ['MAE', 'RMSE', 'R²']
    num_metrics = len(metrics_to_plot)
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, num_metrics, i+1)
        
        if metric == 'R²':
            sorted_df = summary_df.sort_values(metric, ascending=False)
        else:
            sorted_df = summary_df.sort_values(metric, ascending=True)
        
        bars = plt.bar(sorted_df['Category'], sorted_df[metric])
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom', rotation=0)
        
        plt.title(f'{metric} by Category')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_results', f'{adsorbate_type}_category_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models for adsorption energy prediction")
    parser.add_argument("--train", type=str, default="all", 
                        choices=["all", "pairs", "triplets", "fives"],
                        help="Configuration type for training")
    parser.add_argument("--test", type=str, default=None, 
                        choices=["all", "pairs", "triplets", "fives"],
                        help="Configuration type for testing (defaults to same as training)")
    parser.add_argument("--adsorbate", type=str, default="both", 
                        choices=["both", "H", "S"],
                        help="Adsorbate type to train for")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and only run testing on existing model")
    args = parser.parse_args()
    
    base_data_dirs = {
    'pairs': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/pairs',
    'triplets': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/triplets',
    'fives': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/fives'
    }
    
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    if args.adsorbate in ["both", "H"]:
        _, h_results = train_with_configuration(
            train_config=args.train,
            test_configs=args.test,
            adsorbate_type="H", 
            base_data_dirs=base_data_dirs,
            output_dir=model_dir,
            skip_training=args.skip_training
        )
        
    if args.adsorbate in ["both", "S"]:
        _, s_results = train_with_configuration(
            train_config=args.train,
            test_configs=args.test,
            adsorbate_type="S", 
            base_data_dirs=base_data_dirs,
            output_dir=model_dir,
            skip_training=args.skip_training
        )
    
    print("\nProcess complete!")