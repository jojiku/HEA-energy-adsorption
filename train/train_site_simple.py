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
    
    if len(train_val_graphs) == 0: # Handle case where train_val_graphs becomes empty
        return [], [], test_graphs

    val_size_adjusted = val_size / (1 - test_size)
    if val_size_adjusted >= 1.0 and len(train_val_graphs) > 0: # Ensure val_size_adjusted is less than 1
        val_size_adjusted = 0.5 # Or some other reasonable default if val_size makes it too large
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
    
    min_val = min(min(targets), min(predictions)) if len(targets) > 0 and len(predictions) > 0 else 0
    max_val = max(max(targets), max(predictions)) if len(targets) > 0 and len(predictions) > 0 else 0
    
    buffer = (max_val - min_val) * 0.05
    ax.plot([min_val-buffer, max_val+buffer], [min_val-buffer, max_val+buffer], 'r--', linewidth=2)
    
    ax.set_xlabel('DFT Energy (eV)', fontsize=14)
    ax.set_ylabel('Predicted Energy (eV)', fontsize=14)
    ax.set_title(f'{adsorbate_type} Adsorption Energy - Test Set ({data_dir})', fontsize=16)
    
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
    plt.savefig(f'{filename}_parity.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def analyze_metal_effects(test_graphs, predictions, targets, adsorbate_type, data_dir):
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
                
                if metal_symbol not in ['H', 'S']: # Consider only HEA component metals
                    aoi_metals_in_graph.add(metal_symbol)
        
        for metal in aoi_metals_in_graph:
            metal_effects[metal]['dft_energies'].append(dft_energy)
            metal_effects[metal]['pred_energies'].append(pred_energy)

    if not metal_effects:
        print("No metal effects data collected.")
        return

    print(f"\n--- Metal Effects Analysis for {adsorbate_type} on {data_dir} ---")
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


def test_model(model, test_graphs, adsorbate_type, data_dir, filename):
    if len(test_graphs) == 0:
        print("No test graphs available!")
        return {}, [], []
    
    print(f"\n=== Testing {adsorbate_type} model on {len(test_graphs)} graphs ===")
    
    # Ensure test_loader has a consistent batch size, e.g., length of test_graphs or a fixed moderate size
    test_batch_size = len(test_graphs) if len(test_graphs) > 0 else 1 # Avoid batch size 0
    if len(test_graphs) == 0: # Should be caught by above, but defensive
        return {}, [], []

    test_loader = DataLoader(test_graphs, batch_size=test_batch_size, shuffle=False)
    
    model.eval()
    # Assuming model.test can handle the loader directly
    # If model.test expects a list of graphs and a size, adjust accordingly
    # For now, assuming it takes loader and total_size
    predictions, targets, _ = model.test(test_loader, len(test_graphs)) 
    
    metrics = evaluate_metrics(predictions, targets)
    
    print(f"Test Results for {adsorbate_type} on {data_dir}:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    create_parity_plot(predictions, targets, adsorbate_type, data_dir, filename)
    
    # Analyze metal effects
    analyze_metal_effects(test_graphs, predictions, targets, adsorbate_type, data_dir)
    
    return metrics, predictions, targets

def train_simplified(adsorbate_type='H', data_dir='fives', random_state=42):
    base_data_dirs = {
        'pairs': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/pairs',
        'triplets': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/triplets',
        'fives': 'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs_site_specific/fives'
    }
    
    all_graphs = load_site_specific_graphs(base_data_dirs[data_dir], adsorbate_type)
    print(f"Loaded {len(all_graphs)} {adsorbate_type} graphs from {data_dir}")
    
    if len(all_graphs) == 0:
        print("No graphs to train on!")
        return None, None, {} # Return structure consistent with successful run
    
    train_graphs, val_graphs, test_graphs = split_data_sklearn(
        all_graphs, 
        test_size=0.15, 
        val_size=0.15, 
        random_state=random_state
    )
    
    print(f"Split: Train: {len(train_graphs)}, Validation: {len(val_graphs)}, Test: {len(test_graphs)}")

    if not train_graphs:
        print("No training graphs after split. Aborting training.")
        return None, None, {}
    if not val_graphs:
        print("No validation graphs after split. Aborting training.") # Or proceed without validation if intended
        # For this script, let's assume validation is required
        return None, None, {}


    filename = f'lGNN_{adsorbate_type}_{data_dir}'
    batch_size = 32 # Ensure batch_size is not greater than num_train_samples
    if len(train_graphs) < batch_size :
        batch_size = len(train_graphs)
        print(f"Adjusted batch size to {batch_size} due to small training set size.")


    max_epochs = 1000
    learning_rate = 0.001
    
    roll_val_width = 30
    patience = 150
    report_every = 25
    
    input_dim = train_graphs[0].x.shape[1]
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
    
    train_loader_batch_size = batch_size if len(train_graphs) >= batch_size else len(train_graphs)
    val_loader_batch_size = len(val_graphs) if len(val_graphs) > 0 else 1

    train_loader = DataLoader(train_graphs, batch_size=train_loader_batch_size, drop_last=True if len(train_graphs) > train_loader_batch_size else False, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=val_loader_batch_size, drop_last=False, shuffle=False) # drop_last typically False for val/test
    
    train_loss_vals, val_loss_vals = [], [] # Renamed to avoid conflict
    model_states = []
    
    print(f"\n=== Starting training for {adsorbate_type} on {data_dir} ===")
    
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(max_epochs):
        current_train_loss = model.train4epoch(train_loader, train_loader_batch_size, opt) # Pass correct batch size
        train_loss_vals.append(current_train_loss)
        
        if val_graphs: # Only validate if there's validation data
            pred, target, _ = model.test(val_loader, len(val_graphs))
            val_mae = abs(np.array(pred) - np.array(target)).mean() if len(pred) > 0 else float('inf')
            val_loss_vals.append(val_mae)
            model_states.append(deepcopy(model.state_dict()))
            scheduler.step(val_mae)

            if val_mae < best_val_loss:
                best_val_loss = val_mae
                best_epoch = epoch
                # Storing best state directly might be better if early stopping is aggressive
                # best_model_state_for_early_stop = deepcopy(model.state_dict())


            if epoch >= roll_val_width + patience and len(val_loss_vals) > roll_val_width + patience : # Check length of val_loss_vals
                # Ensure roll_val_width+1 is not zero and roll_val is not empty
                if roll_val_width + 1 > 0:
                    roll_val = np.convolve(val_loss_vals, np.ones(int(roll_val_width+1)), 'valid') / int(roll_val_width+1)
                    if len(roll_val) > patience: # Ensure there are enough elements for comparison
                        min_roll_val = np.min(roll_val[:-(patience-1)] if patience > 1 else roll_val) # Corrected slicing
                        current_avg_roll_val = roll_val[-1]
                        
                        # Check if min_roll_val is zero to avoid division by zero
                        if min_roll_val != 0:
                           improv = (current_avg_roll_val - min_roll_val) / min_roll_val
                        else:
                           improv = -1.0 # Assume improvement if min_roll_val is zero and current is also zero, or negative if current is positive

                        if improv > -0.005 : # If improvement is not significant
                            print(f'Early stopping invoked at epoch {epoch}.')
                            # best_epoch determined by overall min val_loss, not necessarily current roll_val state
                            # We need to find the epoch that had the actual minimum validation loss
                            if model_states: # Ensure model_states is not empty
                                actual_best_epoch = np.argmin(val_loss_vals)
                                best_state = model_states[actual_best_epoch]
                                model.load_state_dict(best_state) # Load the true best state
                                print(f'Best validation L1Loss {val_loss_vals[actual_best_epoch]:.3f} eV at epoch {actual_best_epoch}.')

                            break # Exit training loop
        else: # No validation data
            val_loss_vals.append(float('inf')) # Append dummy value or handle differently
            model_states.append(deepcopy(model.state_dict())) # Still save model state

        if epoch % report_every == 0:
            val_report_loss = val_loss_vals[-1] if val_loss_vals else float('nan')
            print(f'Epoch {epoch} train L1Loss: {train_loss_vals[-1]:.3f} / val L1Loss: {val_report_loss:.3f} eV')
        
        if epoch == max_epochs -1: # If loop finishes without early stopping
            if val_loss_vals and model_states:
                actual_best_epoch = np.argmin(val_loss_vals)
                best_state = model_states[actual_best_epoch]
                model.load_state_dict(best_state)
                print(f'Finished training. Best validation L1Loss {val_loss_vals[actual_best_epoch]:.3f} eV at epoch {actual_best_epoch}.')
            elif model_states : # No validation, take last state
                best_state = model_states[-1]
                model.load_state_dict(best_state)
                print(f'Finished training. No validation data. Using model from last epoch.')
            else: # Should not happen if training ran
                print("Warning: No model state saved.")
                return model, None, {}


    # Determine best state if not set by early stopping logic correctly
    if 'best_state' not in locals() and model_states:
        if val_loss_vals:
            final_best_epoch = np.argmin(val_loss_vals)
            best_state = model_states[final_best_epoch]
            print(f'Finalizing: Best epoch was {final_best_epoch} with val. L1Loss {np.min(val_loss_vals):.3f} eV')
        else: # No validation, use last state
            best_state = model_states[-1]
            final_best_epoch = len(model_states) -1
            print(f'Finalizing: No validation data. Using model from last epoch {final_best_epoch}.')
        model.load_state_dict(best_state)
    elif 'best_state' not in locals() and not model_states:
        print("Error: No model states were saved during training.")
        return model, None, {}


    # Save best model state
    if 'best_state' in locals() and best_state is not None:
        best_state_to_save = deepcopy(best_state) # Ensure it's a copy
        best_state_to_save['onehot_labels'] = train_graphs[0].onehot_labels
        best_state_to_save['arch'] = arch
        torch.save(best_state_to_save, f'{filename}.state')
    else:
        print("Warning: best_state not defined, model not saved.")
        return model, None, {}


    # Plot training curve
    if train_loss_vals: # Check if lists are populated
        fig, main_ax = plt.subplots(1, 1, figsize=(8, 5))
        color = ['steelblue','green']
        label = [r'Training set L1Loss',r'Validation set L1Loss']

        main_ax.plot(range(len(train_loss_vals)), train_loss_vals, color=color[0], label=label[0])
        if val_loss_vals and any(vl != float('inf') for vl in val_loss_vals): # Check if val_loss has valid data
            main_ax.plot(range(len(val_loss_vals)), val_loss_vals, color=color[1], label=label[1])
            if 'actual_best_epoch' in locals() and actual_best_epoch < len(val_loss_vals):
                 main_ax.scatter(actual_best_epoch, val_loss_vals[actual_best_epoch], facecolors='none', 
                              edgecolors='maroon', label='Best epoch', s=50, zorder=10)
            elif 'best_epoch' in locals() and best_epoch != -1 and best_epoch < len(val_loss_vals): # Fallback if actual_best_epoch not set
                 main_ax.scatter(best_epoch, val_loss_vals[best_epoch], facecolors='none', 
                              edgecolors='maroon', label='Best epoch (val_mae based)', s=50, zorder=10)


        main_ax.set_xlabel(r'Epoch', fontsize=16)
        main_ax.set_ylabel(r'L1Loss [eV]', fontsize=16)
        main_ax.legend()
        main_ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{filename}_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
    # Test the model
    test_metrics, test_predictions, test_targets = test_model(model, test_graphs, adsorbate_type, data_dir, filename)
    
    return model, best_state if 'best_state' in locals() else None, test_metrics


if __name__ == "__main__":
    # Train and test for H on fives
    print("=" * 60)
    print("TRAINING AND TESTING H ADSORPTION MODEL")
    print("=" * 60)
    model_H, state_H, metrics_H = train_simplified(adsorbate_type='H', data_dir='fives', random_state=42)
    
    print("\n" + "=" * 60)
    print("TRAINING AND TESTING S ADSORPTION MODEL")
    print("=" * 60)
    # Train and test for S on fives
    model_S, state_S, metrics_S = train_simplified(adsorbate_type='S', data_dir='fives', random_state=42)
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    if metrics_H:
      print(f"H Model Test MAE: {metrics_H.get('MAE', float('nan')):.4f} eV, R²: {metrics_H.get('R²', float('nan')):.4f}")
    else:
      print("H Model testing did not produce metrics.")
    if metrics_S:
      print(f"S Model Test MAE: {metrics_S.get('MAE', float('nan')):.4f} eV, R²: {metrics_S.get('R²', float('nan')):.4f}")
    else:
      print("S Model testing did not produce metrics.")
    print("Training and testing complete!")