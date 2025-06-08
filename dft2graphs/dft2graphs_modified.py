import os
import json
import torch
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict, Counter, deque
import torch
from tqdm import tqdm
from ase import Atoms
from ase.neighborlist import build_neighbor_list, natural_cutoffs
from torch_geometric.data import Data
from copy import deepcopy
from itertools import combinations
import ase.build

def read_xyz_with_specific_adsorbate(xyz_file, adsorbate_type, site_index):
    """Read XYZ file and add specific adsorbate at given site"""
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    
    positions = []
    symbols = []
    tags = []
    
    adsorbate_lines = []
    for i in range(2, 2 + n_atoms):
        line = lines[i].strip().split()
        symbol = line[0]
        
        if symbol.startswith('!'):
            clean_symbol = symbol[1:]
            if clean_symbol == adsorbate_type:
                adsorbate_lines.append((i, line))
            continue
        
        x, y, z = float(line[1]), float(line[2]), float(line[3])
        symbols.append(symbol)
        positions.append([x, y, z])
        tags.append(1)  # Start with surface tag
    
    # Add specific adsorbate
    if adsorbate_lines and 0 <= site_index < len(adsorbate_lines):
        i, line = adsorbate_lines[site_index]
        clean_symbol = line[0][1:]
        x, y, z = float(line[1]), float(line[2]), float(line[3])
        symbols.append(clean_symbol)
        positions.append([x, y, z])
        tags.append(0)  # Adsorbate tag
    
    atoms = Atoms(symbols=symbols, positions=positions, tags=tags)
    
    # Set a reasonable default cell if none exists
    if atoms.cell is None or np.allclose(atoms.cell, 0):
        pos = atoms.get_positions()
        min_pos = pos.min(axis=0)
        max_pos = pos.max(axis=0)
        cell_size = max_pos - min_pos + 10.0  # Add 10Å padding
        atoms.set_cell([cell_size[0], cell_size[1], cell_size[2]])
        atoms.set_pbc([True, True, False])  # Periodic in x,y but not z
    
    return atoms

def identify_surface_layers(atoms, z_tolerance=0.5):
    """Identify surface layers based on Z-coordinates and assign proper tags"""
    positions = atoms.get_positions()
    tags = atoms.get_tags()
    
    # Get non-adsorbate atoms
    non_ads_indices = np.where(tags != 0)[0]
    if len(non_ads_indices) == 0:
        return atoms
        
    non_ads_z = positions[non_ads_indices, 2]
    
    # Handle case where all atoms are at the same z-level
    if len(np.unique(np.round(non_ads_z / z_tolerance) * z_tolerance)) == 1:
        # All atoms are in the same layer, assign them as surface
        for idx in non_ads_indices:
            tags[idx] = 1
        atoms.set_tags(tags)
        return atoms
    
    sorted_z = np.sort(np.unique(np.round(non_ads_z / z_tolerance) * z_tolerance))
    
    # Assign tags: highest Z = surface (1), then subsurface (2), etc.
    layer_dict = {}
    for i, z in enumerate(reversed(sorted_z)):
        layer_dict[z] = i + 1
    
    for idx in non_ads_indices:
        z_val = np.round(positions[idx, 2] / z_tolerance) * z_tolerance
        tags[idx] = layer_dict[z_val]
    
    atoms.set_tags(tags)
    return atoms

def ase2ocp_tags(atoms):
    """Convert ASE tag format to OCP tag format"""
    atoms.set_tags([0 if t >= 2 else 2 if t == 0 else 1 for t in atoms.get_tags()])
    return atoms

def get_ensemble_hea(atoms):
    """
    Get ensemble for HEA structures - adapted from original get_ensemble
    Returns ensemble composition, ids, and estimated site type
    """
    atoms = deepcopy(atoms)
    if np.any(np.isin([3,4,5], atoms.get_tags())):
        atoms = ase2ocp_tags(atoms)
    
    # Build neighborlist
    try:
        nl = build_neighbor_list(atoms, cutoffs=natural_cutoffs(atoms, mult=1.1), 
                                self_interaction=False, bothways=True)
    except:
        # Fallback with manual cutoffs if natural_cutoffs fails
        cutoffs = {atom.symbol: 3.0 for atom in atoms}  # 3Å default cutoff
        nl = build_neighbor_list(atoms, cutoffs=cutoffs, 
                                self_interaction=False, bothways=True)
    
    # Find adsorbate
    ads_ids = [a.index for a in atoms if a.tag == 2]  # OCP format: 2 = adsorbate
    if not ads_ids:
        raise Exception("No adsorbate found.")
    
    ads_neighbors = np.array([i for i in nl.get_neighbors(ads_ids[0])[0] if i not in ads_ids])
    if len(ads_neighbors) == 0:
        raise Exception("Adsorbate has no neighboring atoms.")
    
    # Get three nearest surface atoms
    dist = atoms.get_distances(ads_ids[0], ads_neighbors)
    ens_ids = ads_neighbors[np.argsort(dist)][:min(3, len(ads_neighbors))]
    
    # Determine closest ensemble configuration
    pos = atoms.get_positions()
    ens_configs = [[i] for i in ens_ids]
    if len(ens_ids) >= 2:
        ens_configs.extend([[*i] for i in combinations(ens_ids, 2)])
    if len(ens_ids) >= 3:
        ens_configs.append(list(ens_ids))
    
    distances = []
    for e in ens_configs:
        mean = np.mean(pos[e], axis=0)
        delta = pos[ads_ids[0]][:2] - mean[:2]
        distances.append(np.sqrt(np.sum(delta**2)))
    
    closest_ens = ens_configs[np.argsort(distances)[0]]
    
    # Categorize site
    if len(closest_ens) == 1:
        site = 'ontop'
    elif len(closest_ens) == 2:
        site = 'bridge_0'  # Simplified bridge classification
    elif len(closest_ens) == 3:
        site = 'fcc'  # Simplified hollow site classification
    
    # Get ensemble composition
    ensemble = np.array(atoms.get_chemical_symbols())[closest_ens]
    ensemble = dict(Counter(ensemble))
    
    return ensemble, closest_ens, site

def get_cell_lengths(cell):
    """Safely get cell lengths from cell object or array"""
    if hasattr(cell, 'lengths'):
        return cell.lengths()
    elif isinstance(cell, np.ndarray):
        if cell.shape == (3, 3):
            # Calculate lengths from cell matrix
            return np.array([np.linalg.norm(cell[i]) for i in range(3)])
        else:
            return cell
    else:
        return np.array([10.0, 10.0, 10.0])  # Default fallback

def create_hea_template(atoms, target_size=(5, 5, 3)):
    """
    Create standardized template from HEA structure
    Fixed to handle edge cases and prevent infinity errors
    """
    atoms = deepcopy(atoms)
    
    # Convert to OCP tags for processing
    if not np.any(np.isin([0, 1, 2], atoms.get_tags())):
        atoms = ase2ocp_tags(atoms)
    
    # Get original cell and positions
    cell = atoms.get_cell()
    pos = atoms.get_positions()
    
    # Get cell lengths safely
    cell_lengths = get_cell_lengths(cell)
    
    # Check for valid cell
    if np.any(cell_lengths <= 0) or np.any(~np.isfinite(cell_lengths)):
        # Create a default cubic cell if cell is invalid
        max_extent = np.max(pos.max(axis=0) - pos.min(axis=0))
        default_size = max(10.0, max_extent + 5.0)  # Add 5Å padding
        cell_lengths = np.array([default_size, default_size, default_size])
        atoms.set_cell([default_size, default_size, default_size])
    
    # Center the structure
    surface_atoms = [a for a in atoms if a.tag == 1]
    if surface_atoms:
        center = np.mean([a.position for a in surface_atoms], axis=0)
    else:
        center = np.mean(pos, axis=0)
    
    # Get current cell as array for translation
    current_cell = atoms.get_cell()
    if hasattr(current_cell, 'array'):
        cell_center = 0.5 * np.sum(current_cell.array, axis=0)
    else:
        cell_center = 0.5 * np.sum(current_cell, axis=0)
    
    atoms.translate(-center + cell_center)
    atoms.wrap()
    
    # Create target supercell with safe calculations
    target_x, target_y, target_z = target_size
    
    # Use safer approach for repetition factors
    min_cell_size = 3.5  # Minimum cell dimension in Å
    max_rep = 5  # Maximum repetition to prevent huge structures
    
    rep_x = max(1, min(max_rep, int(np.ceil(target_x * min_cell_size / cell_lengths[0]))))
    rep_y = max(1, min(max_rep, int(np.ceil(target_y * min_cell_size / cell_lengths[1]))))
    rep_z = max(1, min(max_rep, int(np.ceil(target_z * min_cell_size / cell_lengths[2]))))
    
    # Create supercell
    try:
        supercell = atoms.repeat((rep_x, rep_y, rep_z))
    except:
        # If repeat fails, just use original structure
        supercell = atoms
        rep_x = rep_y = rep_z = 1
    
    # Create reasonable target cell
    target_length_x = max(min_cell_size * target_x, cell_lengths[0])
    target_length_y = max(min_cell_size * target_y, cell_lengths[1])
    target_length_z = max(min_cell_size * target_z, cell_lengths[2])
    
    target_cell = np.array([
        [target_length_x, 0, 0],
        [0, target_length_y, 0],
        [0, 0, target_length_z]
    ])
    
    supercell.set_cell(target_cell)
    pos = supercell.get_positions()
    
    # Keep atoms within the new cell
    try:
        fractional = np.linalg.solve(target_cell.T, pos.T).T
        inside = np.all((fractional >= -0.001) & (fractional <= 1.001), axis=1)
        template = supercell[inside]
    except:
        # If transformation fails, keep all atoms
        template = supercell
    
    # Ensure we have at least some atoms
    if len(template) == 0:
        template = atoms
    
    # Re-tag layers based on Z-coordinate
    template = identify_surface_layers(template)
    
    return template

def BFS(edges, start_node, max_nodes):
    """Breadth-first search for distance calculation"""
    adjacency_list = defaultdict(list)
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    
    distances = [-1] * max_nodes
    if start_node >= max_nodes or start_node < 0:
        return distances
        
    distances[start_node] = 0
    
    queue = deque([(start_node, 0)])
    visited = {start_node}
    
    while queue:
        current_node, current_distance = queue.popleft()
        
        for neighbor in adjacency_list[current_node]:
            if neighbor < max_nodes and neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = current_distance + 1
                queue.append((neighbor, current_distance + 1))
    
    return distances

def get_atoms_of_interest_hea(atoms, ensemble_ids, site_type):
    """
    Get atoms of interest for HEA structures
    Adapts the original AoI selection to work with variable compositions
    """
    # Convert to OCP tags if needed
    if not np.any(np.isin([0, 1, 2], atoms.get_tags())):
        atoms = ase2ocp_tags(atoms)
    
    # Start with ensemble atoms
    aoi = list(ensemble_ids)
    
    # Add neighboring surface atoms based on site type
    try:
        nl = build_neighbor_list(atoms, cutoffs=natural_cutoffs(atoms, mult=1.1), 
                                self_interaction=False, bothways=True)
    except:
        # Fallback with manual cutoffs
        cutoffs = {atom.symbol: 3.0 for atom in atoms}
        nl = build_neighbor_list(atoms, cutoffs=cutoffs, 
                                self_interaction=False, bothways=True)
    
    if site_type == 'ontop':
        # For ontop: primary binding atom + immediate neighbors
        if ensemble_ids:
            primary = ensemble_ids[0]
            try:
                neighbors = [n for n in nl.get_neighbors(primary)[0] if atoms[n].tag == 1]
                aoi.extend(neighbors[:2])  # Limit to 2 additional atoms
            except:
                pass
                
    elif site_type.startswith('bridge'):
        # For bridge: both binding atoms + their shared neighbors
        for ens_id in ensemble_ids:
            try:
                neighbors = [n for n in nl.get_neighbors(ens_id)[0] if atoms[n].tag == 1]
                aoi.extend(neighbors[:1])  # 1 additional per binding atom
            except:
                pass
                
    elif site_type in ['fcc', 'hcp']:
        # For hollow sites: all three binding atoms + select neighbors
        for ens_id in ensemble_ids:
            try:
                neighbors = [n for n in nl.get_neighbors(ens_id)[0] if atoms[n].tag == 1]
                aoi.extend(neighbors[:1])  # 1 additional per binding atom
            except:
                pass
    
    return sorted(set(aoi))

def atoms2graph_hea(atoms, onehot_labels):
    """
    Convert HEA atoms to graph using standardized template approach
    Combines original methodology with HEA flexibility
    """
    # Get adsorbate type
    ads_atoms = [a for a in atoms if a.tag == 0]
    if not ads_atoms:
        return None
    ads_type = ''.join([a.symbol for a in ads_atoms])
    
    # Identify layers and get ensemble
    atoms = identify_surface_layers(atoms)
    
    try:
        ensemble, ensemble_ids, site_type = get_ensemble_hea(atoms)
    except Exception as e:
        # print(f"Error in ensemble detection: {e}")
        return None
    
    # Create standardized template
    try:
        template = create_hea_template(atoms, target_size=(5, 5, 3))
    except Exception as e:
        print(f"Error in template creation: {e}")
        return None
    
    # Convert back to ASE tags for consistency
    template.set_tags([0 if t == 2 else 1 if t == 1 else t for t in template.get_tags()])
    
    # Build neighbor list and edges
    try:
        nl = build_neighbor_list(template, cutoffs=natural_cutoffs(template, mult=1.1), 
                                self_interaction=False, bothways=True)
    except:
        # Fallback with manual cutoffs
        cutoffs = {atom.symbol: 3.0 for atom in template}
        nl = build_neighbor_list(template, cutoffs=cutoffs, 
                                self_interaction=False, bothways=True)
    
    edges = []
    for a in template:
        try:
            neighbors = nl.get_neighbors(a.index)[0]
            for neighbor in neighbors:
                edges.append([a.index, neighbor])
        except:
            continue
    
    edges = np.array(edges)
    if len(edges) == 0:
        return None
    
    # Find adsorbate in template
    ads_indices = [a.index for a in template if a.tag == 0]
    if not ads_indices:
        return None
    
    # BFS from adsorbate
    edge_dists = BFS(edges, ads_indices[0], len(template))
    
    # Select nodes based on distance and layer
    gIds = ads_indices.copy()
    for t, n in [(0, 1), (1, 2), (2, 2), (3, 3)]:
        for a in template:
            if (a.tag == t and 
                a.index < len(edge_dists) and
                edge_dists[a.index] != -1 and
                edge_dists[a.index] <= n and 
                a.index not in gIds):
                gIds.append(a.index)
    
    gIds = sorted(gIds)
    
    if len(gIds) == 0:
        return None
    
    # Filter edges
    edges = edges[np.all(np.isin(edges, gIds), axis=1)]
    
    # Get atoms of interest using adapted method
    try:
        # Convert template tags back to OCP for AoI calculation
        temp_atoms = deepcopy(template)
        temp_atoms = ase2ocp_tags(temp_atoms)
        temp_ensemble, temp_ens_ids, temp_site = get_ensemble_hea(temp_atoms)
        aoi = get_atoms_of_interest_hea(temp_atoms, temp_ens_ids, temp_site)
        # Convert back to original indexing
        aoi = [i for i in aoi if i in gIds]
    except:
        # Fallback: use ensemble atoms as AoI
        aoi = [i for i in gIds if template[i].tag == 1][:6]  # Limit to 6 atoms
    
    # Create node features
    node_onehot = np.zeros((len(gIds), len(onehot_labels) + 2))
    for i, j in enumerate(gIds):
        symbol = template[j].symbol
        if symbol in onehot_labels:
            node_onehot[i, onehot_labels.index(symbol)] = 1
        node_onehot[i, -2] = template[j].tag
        if j in aoi:
            node_onehot[i, -1] = 1
    
    # Map edges to new indices
    if len(edges) > 0:
        id_map = {g: i for i, g in enumerate(gIds)}
        valid_edges = []
        for edge in edges:
            if edge[0] in id_map and edge[1] in id_map:
                valid_edges.append([id_map[edge[0]], id_map[edge[1]]])
        edges = np.array(valid_edges) if valid_edges else np.array([]).reshape(0, 2)
    else:
        edges = np.array([]).reshape(0, 2)
    
    # Create torch tensors
    if len(edges) > 0:
        torch_edges = torch.tensor(np.transpose(edges), dtype=torch.long)
    else:
        torch_edges = torch.tensor(np.array([[], []]), dtype=torch.long)
    
    torch_nodes = torch.tensor(node_onehot, dtype=torch.float)
    
    # Create graph
    graph = Data(
        x=torch_nodes,
        edge_index=torch_edges,
        onehot_labels=onehot_labels,
        gIds=gIds,
        ads=ads_type,
        ensemble=ensemble,
        site=site_type
    )
    
    return graph

def create_site_specific_graph_standardized(xyz_file, data_file, onehot_labels, adsorbate_type, site_index):
    """Main function to create standardized graph from XYZ file"""
    try:
        atoms = read_xyz_with_specific_adsorbate(xyz_file, adsorbate_type, site_index)
        
        if sum(atoms.get_tags()) == len(atoms):  # No adsorbate
            return None
        
        graph = atoms2graph_hea(atoms, onehot_labels)
        
        if graph is None:
            return None
        
        # Add energy data if available
        if data_file and os.path.exists(data_file):
            with open(data_file, 'r') as f:
                energy_data = json.load(f)
            
            energy_key = f"energies_{adsorbate_type}_ads_raw"
            if energy_key in energy_data and len(energy_data[energy_key]) > site_index:
                energy = energy_data[energy_key][site_index]
                if np.isfinite(energy):  # Check for valid energy
                    graph.y = energy
                else:
                    return None
        
        return graph
        
    except Exception as e:
        # print(f"Error in create_site_specific_graph_standardized: {e}")
        return None

def process_data_directories_standardized(root_dir="C:/Users/Tseh/Documents/Files/HEA/hea_project"):
    """Main processing function with standardized approach"""
    source_dirs = {
        "triplets": os.path.join(root_dir, "train_data_dft/HEA_results_fives"),
    }
    
    output_dir = os.path.join(root_dir, "train_data_graphs/graphs_standardized")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all elements
    all_elements = set()
    print("Scanning elements...")
    for category, source_dir in source_dirs.items():
        if os.path.exists(source_dir):
            for alloy_dir in os.listdir(source_dir):
                alloy_path = os.path.join(source_dir, alloy_dir)
                if os.path.isdir(alloy_path):
                    data_file = os.path.join(alloy_path, "data.json")
                    if os.path.exists(data_file):
                        with open(data_file, 'r') as f:
                            data = json.load(f)
                            if "composition" in data:
                                all_elements.update(data["composition"])
    
    all_elements.update(["H", "S"])
    onehot_labels = sorted(list(all_elements))
    print(f"Found elements: {onehot_labels}")
    
    # Collect all directories
    all_alloy_dirs = []
    for category, source_dir in source_dirs.items():
        if os.path.exists(source_dir):
            for alloy_dir in os.listdir(source_dir):
                alloy_path = os.path.join(source_dir, alloy_dir)
                if os.path.isdir(alloy_path):
                    xyz_file = os.path.join(alloy_path, "geometry.xyz")
                    data_file = os.path.join(alloy_path, "data.json")
                    if os.path.exists(xyz_file) and os.path.exists(data_file):
                        all_alloy_dirs.append((category, alloy_dir, alloy_path))
    
    # Split data
    np.random.seed(42)
    np.random.shuffle(all_alloy_dirs)
    
    n_total = len(all_alloy_dirs)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    splits = {
        'train': all_alloy_dirs[:n_train],
        'val': all_alloy_dirs[n_train:n_train+n_val],
        'test': all_alloy_dirs[n_train+n_val:]
    }
    
    # Process each split
    for split_name, alloy_list in splits.items():
        print(f"\nProcessing {split_name} set ({len(alloy_list)} alloys)...")
        graph_list = []
        
        for category, alloy_dir, alloy_path in tqdm(alloy_list):
            xyz_file = os.path.join(alloy_path, "geometry.xyz")
            data_file = os.path.join(alloy_path, "data.json")
            
            for adsorbate_type in ['H', 'S']:
                for site_index in range(9):
                    try:
                        graph = create_site_specific_graph_standardized(
                            xyz_file, data_file, onehot_labels, adsorbate_type, site_index
                        )
                        
                        if graph is not None and hasattr(graph, 'y'):
                            graph_list.append(graph)
                    except Exception as e:
                        # More specific error handling
                        if "infinity" in str(e).lower():
                            print(f"Infinity error in {alloy_path}, {adsorbate_type}, site {site_index}")
                        elif "lengths" in str(e).lower():
                            print(f"Cell lengths error in {alloy_path}, {adsorbate_type}, site {site_index}")
                        # Uncomment for debugging:
                        # else:
                        #     print(f"Error processing {alloy_path}, {adsorbate_type}, site {site_index}: {str(e)}")
                        continue
        
        output_path = os.path.join(output_dir, f"{split_name}.graphs")
        with open(output_path, 'wb') as output:
            pickle.dump(graph_list, output)
        
        print(f"Saved {len(graph_list)} graphs to {output_path}")
    
    print("\nStandardized processing completed!")

if __name__ == "__main__":
    process_data_directories_standardized()