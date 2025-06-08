import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
from cheatools.lgnn import lGNN # Assuming cheatools.lgnn is available in your environment
from copy import deepcopy
from collections import defaultdict

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
    if len(graphs) == 0:
        return [], [], []
    
    train_val_graphs, test_graphs = train_test_split(
        graphs, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    if len(train_val_graphs) == 0:
        return [], [], test_graphs

    val_size_adjusted = val_size / (1 - test_size)
    if val_size_adjusted >= 1.0 and len(train_val_graphs) > 0:
        val_size_adjusted = 0.5
        print(f"Warning: Adjusted validation size was too large. Setting to {val_size_adjusted}")
    elif len(train_val_graphs) == 0:
         return [], [], test_graphs

    train_graphs, val_graphs = train_test_split(
        train_val_graphs,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True
    )
    
    return train_graphs, val_graphs, test_graphs

def sample_training_data(train_graphs, percentage, random_state=42):
    """Sample a percentage of training data while maintaining reproducibility"""
    if percentage >= 100:
        return train_graphs
    
    np.random.seed(random_state)
    n_samples = int(len(train_graphs) * percentage / 100)
    n_samples = max(1, n_samples)  # Ensure at least 1 sample
    
    indices = np.random.choice(len(train_graphs), size=n_samples, replace=False)
    sampled_graphs = [train_graphs[i] for i in indices]
    
    return sampled_graphs

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

def create_parity_plot(predictions, targets, adsorbate_type, data_dir, filename, percentage=100):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(targets, predictions, alpha=0.7, edgecolor='black')
    
    min_val = min(min(targets), min(predictions)) if len(targets) > 0 and len(predictions) > 0 else 0
    max_val = max(max(targets), max(predictions)) if len(targets) > 0 and len(predictions) > 0 else 0
    
    buffer = (max_val - min_val) * 0.05
    ax.plot([min_val-buffer, max_val+buffer], [min_val-buffer, max_val+buffer], 'r--', linewidth=2)
    
    ax.set_xlabel('DFT Energy (eV)', fontsize=14)
    ax.set_ylabel('Predicted Energy (eV)', fontsize=14)
    ax.set_title(f'{adsorbate_type} Adsorption Energy - Test Set ({data_dir}, {percentage}% train data)', fontsize=16)
    
    if len(targets) > 0 and len(predictions) > 0:
        metrics = evaluate_metrics(predictions, targets)
        metrics_text = f"MAE: {metrics['MAE']:.3f} eV\nRMSE: {metrics['RMSE']:.3f} eV\nR²: {metrics['R²']:.3f}"
        ax.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    va='top', ha='left', fontsize=12)
    
    ax.axis('equal')
    ax.axis('square')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{filename}_parity_{percentage}pct.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def analyze_metal_effects(test_graphs, predictions, targets, adsorbate_type, data_dir, percentage=100):
    if not test_graphs:
        print("No test graphs to analyze metal effects.")
        return

    onehot_labels = test_graphs[0].onehot_labels
    
    metal_effects = defaultdict(lambda: {'dft_energies': [], 'pred_energies': []})

    for i, graph in enumerate(test_graphs):
        pred_energy = predictions[i]
        dft_energy = targets[i]
        
        aoi_metals_in_graph = set()
        
        for node_idx in range(graph.x.shape[0]):
            node_features = graph.x[node_idx]
            is_in_aoi = node_features[-1].item() == 1
            
            if is_in_aoi:
                element_onehot = node_features[:-2] 
                element_idx = torch.argmax(element_onehot).item()
                metal_symbol = onehot_labels[element_idx]
                
                if metal_symbol not in ['H', 'S']:
                    aoi_metals_in_graph.add(metal_symbol)
        
        for metal in aoi_metals_in_graph:
            metal_effects[metal]['dft_energies'].append(dft_energy)
            metal_effects[metal]['pred_energies'].append(pred_energy)

    if not metal_effects:
        print("No metal effects data collected.")
        return

    print(f"\n--- Metal Effects Analysis for {adsorbate_type} on {data_dir} ({percentage}% train data) ---")
    print(f"{'Metal':<8} | {'Count':<7} | {'Avg DFT E (eV)':<16} | {'Avg Pred E (eV)':<17} | {'MAE (eV)':<10}")
    print("-" * 70)

    sorted_metals = sorted(metal_effects.keys())

    for metal in sorted_metals:
        data = metal_effects[metal]
        dft_e = np.array(data['dft_energies'])
        pred_e = np.array(data['pred_energies'])
        
        count = len(dft_e)
        avg_dft_e = np.mean(dft_e) if count > 0 else float('nan')
        avg_pred_e = np.mean(pred_e) if count > 0 else float('nan')
        mae_metal = mean_absolute_error(dft_e, pred_e) if count > 0 else float('nan')
        
        print(f"{metal:<8} | {count:<7} | {avg_dft_e:<16.3f} | {avg_pred_e:<17.3f} | {mae_metal:<10.3f}")
    print("-" * 70)

def test_model(model, test_graphs, adsorbate_type, data_dir, filename, percentage=100):
    if len(test_graphs) == 0:
        print("No test graphs available!")
        return {}, [], []
    
    print(f"\n=== Testing {adsorbate_type} model on {len(test_graphs)} graphs ({percentage}% train data) ===")
    
    test_batch_size = len(test_graphs) if len(test_graphs) > 0 else 1
    if len(test_graphs) == 0:
        return {}, [], []

    test_loader = DataLoader(test_graphs, batch_size=test_batch_size, shuffle=False)
    
    model.eval()
    predictions, targets, _ = model.test(test_loader, len(test_graphs)) 
    
    metrics = evaluate_metrics(predictions, targets)
    
    print(f"Test Results for {adsorbate_type} on {data_dir} ({percentage}% train data):")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    create_parity_plot(predictions, targets, adsorbate_type, data_dir, filename, percentage)
    analyze_metal_effects(test_graphs, predictions, targets, adsorbate_type, data_dir, percentage)
    
    return metrics, predictions, targets

def train_with_percentage(train_graphs, val_graphs, test_graphs, adsorbate_type, data_dir, percentage, random_state=42):
    """Train model with a specific percentage of training data"""
    
    # Sample training data based on percentage
    sampled_train_graphs = sample_training_data(train_graphs, percentage, random_state)
    
    print(f"\n=== Training with {percentage}% of data ({len(sampled_train_graphs)}/{len(train_graphs)} samples) ===")
    
    if not sampled_train_graphs:
        print("No training graphs after sampling. Aborting training.")
        return None, None, {}
    
    filename = f'lGNN_{adsorbate_type}_{data_dir}_{percentage}pct'
    batch_size = 32
    if len(sampled_train_graphs) < batch_size:
        batch_size = len(sampled_train_graphs)
        print(f"Adjusted batch size to {batch_size} due to small training set size.")

    max_epochs = 1000
    learning_rate = 0.001
    
    roll_val_width = 30
    patience = 150
    report_every = 25
    
    input_dim = sampled_train_graphs[0].x.shape[1]
    arch = {
        'n_conv_layers': 3,
        'n_hidden_layers': 1,
        'conv_dim': 64,
        'input_dim': input_dim,
        'act': 'relu',
        'harmonic': True,
    }

    model = lGNN(arch=arch)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.7, patience=50, verbose=False)
    
    train_loader_batch_size = batch_size if len(sampled_train_graphs) >= batch_size else len(sampled_train_graphs)
    val_loader_batch_size = len(val_graphs) if len(val_graphs) > 0 else 1

    train_loader = DataLoader(sampled_train_graphs, batch_size=train_loader_batch_size, 
                             drop_last=True if len(sampled_train_graphs) > train_loader_batch_size else False, 
                             shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=val_loader_batch_size, drop_last=False, shuffle=False)
    
    train_loss_vals, val_loss_vals = [], []
    model_states = []
    
    print(f"\n=== Starting training for {adsorbate_type} on {data_dir} ({percentage}% data) ===")
    
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(max_epochs):
        current_train_loss = model.train4epoch(train_loader, train_loader_batch_size, opt)
        train_loss_vals.append(current_train_loss)
        
        if val_graphs:
            pred, target, _ = model.test(val_loader, len(val_graphs))
            val_mae = abs(np.array(pred) - np.array(target)).mean() if len(pred) > 0 else float('inf')
            val_loss_vals.append(val_mae)
            model_states.append(deepcopy(model.state_dict()))
            scheduler.step(val_mae)

            if val_mae < best_val_loss:
                best_val_loss = val_mae
                best_epoch = epoch

            if epoch >= roll_val_width + patience and len(val_loss_vals) > roll_val_width + patience:
                if roll_val_width + 1 > 0:
                    roll_val = np.convolve(val_loss_vals, np.ones(int(roll_val_width+1)), 'valid') / int(roll_val_width+1)
                    if len(roll_val) > patience:
                        min_roll_val = np.min(roll_val[:-(patience-1)] if patience > 1 else roll_val)
                        current_avg_roll_val = roll_val[-1]
                        
                        if min_roll_val != 0:
                           improv = (current_avg_roll_val - min_roll_val) / min_roll_val
                        else:
                           improv = -1.0

                        if improv > -0.005:
                            print(f'Early stopping invoked at epoch {epoch}.')
                            if model_states:
                                actual_best_epoch = np.argmin(val_loss_vals)
                                best_state = model_states[actual_best_epoch]
                                model.load_state_dict(best_state)
                                print(f'Best validation L1Loss {val_loss_vals[actual_best_epoch]:.3f} eV at epoch {actual_best_epoch}.')
                            break
        else:
            val_loss_vals.append(float('inf'))
            model_states.append(deepcopy(model.state_dict()))

        if epoch % report_every == 0:
            val_report_loss = val_loss_vals[-1] if val_loss_vals else float('nan')
            print(f'Epoch {epoch} train L1Loss: {train_loss_vals[-1]:.3f} / val L1Loss: {val_report_loss:.3f} eV')
        
        if epoch == max_epochs - 1:
            if val_loss_vals and model_states:
                actual_best_epoch = np.argmin(val_loss_vals)
                best_state = model_states[actual_best_epoch]
                model.load_state_dict(best_state)
                print(f'Finished training. Best validation L1Loss {val_loss_vals[actual_best_epoch]:.3f} eV at epoch {actual_best_epoch}.')
            elif model_states:
                best_state = model_states[-1]
                model.load_state_dict(best_state)
                print(f'Finished training. No validation data. Using model from last epoch.')
            else:
                print("Warning: No model state saved.")
                return model, None, {}

    # Finalize best state
    if 'best_state' not in locals() and model_states:
        if val_loss_vals:
            final_best_epoch = np.argmin(val_loss_vals)
            best_state = model_states[final_best_epoch]
            print(f'Finalizing: Best epoch was {final_best_epoch} with val. L1Loss {np.min(val_loss_vals):.3f} eV')
        else:
            best_state = model_states[-1]
            final_best_epoch = len(model_states) - 1
            print(f'Finalizing: No validation data. Using model from last epoch {final_best_epoch}.')
        model.load_state_dict(best_state)
    elif 'best_state' not in locals() and not model_states:
        print("Error: No model states were saved during training.")
        return model, None, {}

    # Save best model state
    if 'best_state' in locals() and best_state is not None:
        best_state_to_save = deepcopy(best_state)
        best_state_to_save['onehot_labels'] = sampled_train_graphs[0].onehot_labels
        best_state_to_save['arch'] = arch
        torch.save(best_state_to_save, f'{filename}.state')
    else:
        print("Warning: best_state not defined, model not saved.")
        return model, None, {}

    # Plot training curve
    if train_loss_vals:
        fig, main_ax = plt.subplots(1, 1, figsize=(8, 5))
        color = ['steelblue','green']
        label = [r'Training set L1Loss',r'Validation set L1Loss']

        main_ax.plot(range(len(train_loss_vals)), train_loss_vals, color=color[0], label=label[0])
        if val_loss_vals and any(vl != float('inf') for vl in val_loss_vals):
            main_ax.plot(range(len(val_loss_vals)), val_loss_vals, color=color[1], label=label[1])
            if 'actual_best_epoch' in locals() and actual_best_epoch < len(val_loss_vals):
                 main_ax.scatter(actual_best_epoch, val_loss_vals[actual_best_epoch], facecolors='none', 
                              edgecolors='maroon', label='Best epoch', s=50, zorder=10)
            elif 'best_epoch' in locals() and best_epoch != -1 and best_epoch < len(val_loss_vals):
                 main_ax.scatter(best_epoch, val_loss_vals[best_epoch], facecolors='none', 
                              edgecolors='maroon', label='Best epoch (val_mae based)', s=50, zorder=10)

        main_ax.set_xlabel(r'Epoch', fontsize=16)
        main_ax.set_ylabel(r'L1Loss [eV]', fontsize=16)
        main_ax.set_title(f'{adsorbate_type} Training Curve ({percentage}% data)', fontsize=16)
        main_ax.legend()
        main_ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{filename}_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
    # Test the model
    test_metrics, test_predictions, test_targets = test_model(model, test_graphs, adsorbate_type, data_dir, filename, percentage)
    
    return model, best_state if 'best_state' in locals() else None, test_metrics

def train_data_percentage_analysis(adsorbate_type='H', data_dir='fives', random_state=42):
    """Train models with different percentages of training data and compare results"""
    
    base_data_dirs = {
        'pairs': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/pairs',
        'triplets': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/triplets',
        'fives': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/fives'
    }
    
    all_graphs = load_site_specific_graphs(base_data_dirs[data_dir], adsorbate_type)
    print(f"Loaded {len(all_graphs)} {adsorbate_type} graphs from {data_dir}")
    
    if len(all_graphs) == 0:
        print("No graphs to train on!")
        return {}
    
    # Split data once to keep test and validation sets consistent
    train_graphs, val_graphs, test_graphs = split_data_sklearn(
        all_graphs, 
        test_size=0.15, 
        val_size=0.15, 
        random_state=random_state
    )
    
    print(f"Data split: Train: {len(train_graphs)}, Validation: {len(val_graphs)}, Test: {len(test_graphs)}")

    if not train_graphs or not val_graphs or not test_graphs:
        print("Insufficient data after split. Aborting analysis.")
        return {}

    # Define percentages to test
    percentages = [10, 30, 50, 70, 100]
    results = {}
    
    # Train models with different percentages of training data
    for percentage in percentages:
        print(f"\n{'='*80}")
        print(f"TRAINING WITH {percentage}% OF TRAINING DATA")
        print(f"{'='*80}")
        
        model, state, metrics = train_with_percentage(
            train_graphs, val_graphs, test_graphs, 
            adsorbate_type, data_dir, percentage, random_state
        )
        
        results[percentage] = {
            'model': model,
            'state': state,
            'metrics': metrics,
            'train_samples': len(sample_training_data(train_graphs, percentage, random_state))
        }
    
    # Create comparison plot
    create_comparison_plot(results, adsorbate_type, data_dir)
    
    # Print summary
    print_results_summary(results, adsorbate_type, data_dir)
    
    return results

def create_comparison_plot(results, adsorbate_type, data_dir):
    """Create a plot comparing performance vs training data percentage"""
    
    percentages = sorted(results.keys())
    mae_values = []
    r2_values = []
    train_samples = []
    
    for pct in percentages:
        if results[pct]['metrics']:
            mae_values.append(results[pct]['metrics'].get('MAE', float('nan')))
            r2_values.append(results[pct]['metrics'].get('R²', float('nan')))
            train_samples.append(results[pct]['train_samples'])
        else:
            mae_values.append(float('nan'))
            r2_values.append(float('nan'))
            train_samples.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE vs percentage plot
    ax1.plot(percentages, mae_values, 'o-', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('Training Data Percentage (%)', fontsize=12)
    ax1.set_ylabel('Test MAE (eV)', fontsize=12)
    ax1.set_title(f'{adsorbate_type} Model Performance vs Training Data\n({data_dir})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(percentages)
    
    # Add sample count as secondary x-axis
    ax1_twin = ax1.twiny()
    ax1_twin.set_xlim(ax1.get_xlim())
    ax1_twin.set_xticks(percentages)
    ax1_twin.set_xticklabels([f'{n}' for n in train_samples])
    ax1_twin.set_xlabel('Number of Training Samples', fontsize=12)
    
    # R² vs percentage plot
    ax2.plot(percentages, r2_values, 'o-', linewidth=2, markersize=8, color='blue')
    ax2.set_xlabel('Training Data Percentage (%)', fontsize=12)
    ax2.set_ylabel('Test R²', fontsize=12)
    ax2.set_title(f'{adsorbate_type} Model R² vs Training Data\n({data_dir})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(percentages)
    
    # Add sample count as secondary x-axis
    ax2_twin = ax2.twiny()
    ax2_twin.set_xlim(ax2.get_xlim())
    ax2_twin.set_xticks(percentages)
    ax2_twin.set_xticklabels([f'{n}' for n in train_samples])
    ax2_twin.set_xlabel('Number of Training Samples', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'data_percentage_analysis_{adsorbate_type}_{data_dir}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def print_results_summary(results, adsorbate_type, data_dir):
    """Print a summary table of results"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING DATA PERCENTAGE ANALYSIS SUMMARY - {adsorbate_type} on {data_dir}")
    print(f"{'='*80}")
    print(f"{'Percentage':<12} | {'Samples':<8} | {'MAE (eV)':<10} | {'RMSE (eV)':<11} | {'R²':<8}")
    print("-" * 80)
    
    for percentage in sorted(results.keys()):
        metrics = results[percentage]['metrics']
        samples = results[percentage]['train_samples']
        
        if metrics:
            mae = metrics.get('MAE', float('nan'))
            rmse = metrics.get('RMSE', float('nan'))
            r2 = metrics.get('R²', float('nan'))
            
            print(f"{percentage:>10}% | {samples:<8} | {mae:<10.4f} | {rmse:<11.4f} | {r2:<8.4f}")
        else:
            print(f"{percentage:>10}% | {samples:<8} | {'N/A':<10} | {'N/A':<11} | {'N/A':<8}")
    
    print("-" * 80)
    
    # Find the point of diminishing returns
    valid_results = {k: v for k, v in results.items() if v['metrics']}
    if len(valid_results) >= 2:
        mae_values = [(k, v['metrics']['MAE']) for k, v in valid_results.items()]
        mae_values.sort()
        
        print(f"\nDiminishing Returns Analysis:")
        for i in range(1, len(mae_values)):
            prev_pct, prev_mae = mae_values[i-1]
            curr_pct, curr_mae = mae_values[i]
            improvement = (prev_mae - curr_mae) / prev_mae * 100
            data_increase = (curr_pct - prev_pct)
            efficiency = improvement / data_increase if data_increase > 0 else 0
            
            print(f"  {prev_pct}% → {curr_pct}%: {improvement:.2f}% MAE improvement for {data_increase}% more data (efficiency: {efficiency:.3f})")

if __name__ == "__main__":
    # Analyze H adsorption with different training data percentages
    print("=" * 60)
    print("H ADSORPTION - TRAINING DATA PERCENTAGE ANALYSIS")
    print("=" * 60)
    results_H = train_data_percentage_analysis(adsorbate_type='H', data_dir='fives', random_state=42)
    
    print("\n" + "=" * 60)
    print("S ADSORPTION - TRAINING DATA PERCENTAGE ANALYSIS")
    print("=" * 60)
    # Analyze S adsorption with different training data percentages
    results_S = train_data_percentage_analysis(adsorbate_type='S', data_dir='fives', random_state=42)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Check the generated plots and summary tables to understand")
    print("the relationship between training data size and model performance.")