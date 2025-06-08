import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
from cheatools.lgnn import lGNN
from copy import deepcopy

def load_site_specific_graphs(graph_dir, adsorbate_type='H'):
    all_graphs = []
    
    pattern = os.path.join(graph_dir, f"*_{adsorbate_type}_site*.pt")
    for graph_file in tqdm(glob(pattern), desc=f"Loading {adsorbate_type} graphs"):
        try:
            graph = torch.load(graph_file)
            if hasattr(graph, 'energy'):
                graph.y = graph.energy  
                all_graphs.append(graph)
            else:
                print(f"Warning: {graph_file} has no energy data. Skipping.")
        except Exception as e:
            print(f"Error loading {graph_file}: {str(e)}")
    
    return all_graphs

def split_data_sklearn(graphs, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train/validation/test sets using sklearn's train_test_split
    to ensure no data leakage and proper stratification if needed.
    
    Args:
        graphs: List of graph objects
        test_size: Proportion of data for test set (default: 0.15)
        val_size: Proportion of data for validation set (default: 0.15)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        train_graphs, val_graphs, test_graphs
    """
    if len(graphs) == 0:
        return [], [], []
    
    # First split: separate out test set
    train_val_graphs, test_graphs = train_test_split(
        graphs, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    # Second split: separate train and validation from remaining data
    # Adjust validation size relative to the remaining data
    val_size_adjusted = val_size / (1 - test_size)
    
    train_graphs, val_graphs = train_test_split(
        train_val_graphs,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True
    )
    
    return train_graphs, val_graphs, test_graphs

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
        'Max Error': np.max(np.abs(predictions - targets))
    }

def create_parity_plot(predictions, targets, adsorbate_type, data_dir, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(targets, predictions, alpha=0.7, edgecolor='black')
    
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    buffer = (max_val - min_val) * 0.05
    ax.plot([min_val-buffer, max_val+buffer], [min_val-buffer, max_val+buffer], 'r--', linewidth=2)
    
    ax.set_xlabel('DFT Energy (eV)', fontsize=14)
    ax.set_ylabel('Predicted Energy (eV)', fontsize=14)
    ax.set_title(f'{adsorbate_type} Adsorption Energy - Test Set ({data_dir})', fontsize=16)
    
    metrics = evaluate_metrics(predictions, targets)
    metrics_text = f"MAE: {metrics['MAE']:.3f} eV\nRMSE: {metrics['RMSE']:.3f} eV\nR²: {metrics['R²']:.3f}"
    ax.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                va='top', ha='left', fontsize=12)
    
    ax.axis('equal')
    ax.axis('square')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{filename}_parity.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_model(model, test_graphs, adsorbate_type, data_dir, filename):
    if len(test_graphs) == 0:
        print("No test graphs available!")
        return
    
    print(f"\n=== Testing {adsorbate_type} model on {len(test_graphs)} graphs ===")
    
    test_loader = DataLoader(test_graphs, batch_size=len(test_graphs), shuffle=False)
    
    model.eval()
    predictions, targets, _ = model.test(test_loader, len(test_graphs))
    
    metrics = evaluate_metrics(predictions, targets)
    
    print(f"Test Results for {adsorbate_type} on {data_dir}:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    create_parity_plot(predictions, targets, adsorbate_type, data_dir, filename)
    
    return metrics, predictions, targets

def train_simplified(adsorbate_type='H', data_dir='fives', random_state=42):
    """Simplified training function with testing using sklearn data splitting"""
    
    # Data directories
    base_data_dirs = {
        'pairs': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/pairs',
        'triplets': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/triplets',
        'fives': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/fives'
    }
    
    # Load all graphs
    all_graphs = load_site_specific_graphs(base_data_dirs[data_dir], adsorbate_type)
    print(f"Loaded {len(all_graphs)} {adsorbate_type} graphs from {data_dir}")
    
    if len(all_graphs) == 0:
        print("No graphs to train on!")
        return
    
    # Split data using sklearn (70% train, 15% val, 15% test)
    train_graphs, val_graphs, test_graphs = split_data_sklearn(
        all_graphs, 
        test_size=0.15, 
        val_size=0.15, 
        random_state=random_state
    )
    
    print(f"Split: Train: {len(train_graphs)}, Validation: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Training parameters
    filename = f'lGNN_{adsorbate_type}_{data_dir}'
    batch_size = 32
    max_epochs = 1000
    learning_rate = 0.001
    
    roll_val_width = 30
    patience = 150
    report_every = 25
    
    # Model architecture
    input_dim = train_graphs[0].x.shape[1]
    arch = {
        'n_conv_layers': 3,
        'n_hidden_layers': 1,
        'conv_dim': 64,
        'input_dim': input_dim,
        'act': 'relu',
        'harmonic': True,
    }

    # Initialize model and optimizer
    model = lGNN(arch=arch)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.7, patience=50, verbose=True)
    
    # Data loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=len(val_graphs), drop_last=True, shuffle=False)
    
    # Training loop
    train_loss, val_loss = [], []
    model_states = []
    
    print(f"\n=== Starting training for {adsorbate_type} on {data_dir} ===")
    
    for epoch in range(max_epochs):
        train_loss.append(model.train4epoch(train_loader, batch_size, opt))
        pred, target, _ = model.test(val_loader, len(val_graphs))
        val_mae = abs(np.array(pred) - np.array(target)).mean()
        val_loss.append(val_mae)
        model_states.append(deepcopy(model.state_dict()))
        
        scheduler.step(val_mae)

        # Early stopping
        if epoch >= roll_val_width + patience:
            roll_val = np.convolve(val_loss, np.ones(int(roll_val_width+1)), 'valid') / int(roll_val_width+1)
            min_roll_val = np.min(roll_val[:-patience+1])
            improv = (roll_val[-1] - min_roll_val) / min_roll_val

            if improv > -0.005:
                print('Early stopping invoked.')
                best_epoch = np.argmin(val_loss)
                best_state = model_states[best_epoch]
                break

        if epoch % report_every == 0:
            print(f'Epoch {epoch} train and val L1Loss: {train_loss[-1]:.3f} / {val_loss[-1]:.3f} eV')

    # Load best model
    best_epoch = np.argmin(val_loss)
    best_state = model_states[best_epoch]
    print(f'Finished training. Best epoch was {best_epoch} with val. L1Loss {np.min(val_loss):.3f} eV')
    
    # Load best state into model
    model.load_state_dict(best_state)
    
    # Save best model
    best_state['onehot_labels'] = train_graphs[0].onehot_labels
    best_state['arch'] = arch
    
    torch.save(best_state, f'{filename}.state')
    
    # Plot training curve
    fig, main_ax = plt.subplots(1, 1, figsize=(8, 5))
    color = ['steelblue','green']
    label = [r'Training set L1Loss',r'Validation set L1Loss']

    for i, results in enumerate([train_loss, val_loss]):
        main_ax.plot(range(len(results)), results, color=color[i], label=label[i])
        if i == 1:
            main_ax.scatter(best_epoch, val_loss[best_epoch], facecolors='none', 
                          edgecolors='maroon', label='Best epoch', s=50, zorder=10)

    main_ax.set_xlabel(r'Epoch', fontsize=16)
    main_ax.set_ylabel(r'L1Loss [eV]', fontsize=16)
    main_ax.legend()
    main_ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{filename}_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test the model
    test_metrics, test_predictions, test_targets = test_model(model, test_graphs, adsorbate_type, data_dir, filename)
    
    return model, best_state, test_metrics

if __name__ == "__main__":
    # Train and test for H on fives
    print("=" * 60)
    print("TRAINING AND TESTING H ADSORPTION MODEL")
    print("=" * 60)
    model_H, state_H, metrics_H = train_simplified(adsorbate_type='H', data_dir='fives')
    
    print("\n" + "=" * 60)
    print("TRAINING AND TESTING S ADSORPTION MODEL")
    print("=" * 60)
    # Train and test for S on fives
    model_S, state_S, metrics_S = train_simplified(adsorbate_type='S', data_dir='fives')
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"H Model Test MAE: {metrics_H['MAE']:.4f} eV, R²: {metrics_H['R²']:.4f}")
    print(f"S Model Test MAE: {metrics_S['MAE']:.4f} eV, R²: {metrics_S['R²']:.4f}")
    print("Training and testing complete!")

