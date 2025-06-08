import os
import json
import torch
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import torch
from tqdm import tqdm
from ase import Atoms
from ase.neighborlist import build_neighbor_list, natural_cutoffs
from collections import Counter, deque, defaultdict
from torch_geometric.data import Data

def read_xyz_with_specific_adsorbate(xyz_file, adsorbate_type, site_index):
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
        tags.append(1) 
    
    if adsorbate_lines and 0 <= site_index < len(adsorbate_lines):
        i, line = adsorbate_lines[site_index]
        clean_symbol = line[0][1:]  
        x, y, z = float(line[1]), float(line[2]), float(line[3])
        symbols.append(clean_symbol)
        positions.append([x, y, z])
        tags.append(0)  
    
    atoms = Atoms(symbols=symbols, positions=positions, tags=tags)
    return atoms

def identify_surface_layers(atoms, z_tolerance=0.5):
    positions = atoms.get_positions()
    tags = atoms.get_tags()
    
    non_ads_indices = np.where(tags != 0)[0]
    if len(non_ads_indices) == 0:
        return atoms
        
    non_ads_z = positions[non_ads_indices, 2]
    sorted_z = np.sort(np.unique(np.round(non_ads_z / z_tolerance) * z_tolerance))
    
    layer_dict = {z: i+1 for i, z in enumerate(sorted_z)}
    
    for i, idx in enumerate(non_ads_indices):
        z_val = np.round(positions[idx, 2] / z_tolerance) * z_tolerance
        tags[idx] = layer_dict[z_val]
    
    atoms.set_tags(tags)
    return atoms

def BFS(edges, start_node, max_nodes):
    adjacency_list = defaultdict(list)
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    
    distances = [-1] * max_nodes
    distances[start_node] = 0
    
    queue = deque([(start_node, 0)])
    visited = {start_node}
    
    while queue:
        current_node, current_distance = queue.popleft()
        
        for neighbor in adjacency_list[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = current_distance + 1
                queue.append((neighbor, current_distance + 1))
    
    return distances

def create_site_specific_graph(xyz_file, data_file, onehot_labels, adsorbate_type, site_index):
    atoms = read_xyz_with_specific_adsorbate(xyz_file, adsorbate_type, site_index)
    
    if sum(atoms.get_tags()) == len(atoms):  
        return None
    
    atoms = identify_surface_layers(atoms)
    
    site_type = "ontop" if adsorbate_type == "H" else "triangle"
    binding_atoms = []
    
    if adsorbate_type == "H":
        cutoff_mult = 1.5
    else:
        cutoff_mult = 1.8
    
    nl = build_neighbor_list(atoms, cutoffs=natural_cutoffs(atoms, mult=cutoff_mult), 
                            self_interaction=False, bothways=True)
    
    ads_indices = [a.index for a in atoms if a.tag == 0]
    if not ads_indices:
        return None
    
    for ads_idx in ads_indices:
        neighbors = nl.get_neighbors(ads_idx)[0]
        surface_neighbors = [n for n in neighbors if atoms[n].tag >= 1]
        binding_atoms.extend(surface_neighbors)
            
    binding_atoms = sorted(set(binding_atoms))
    
    aoi = binding_atoms.copy()
    
    for atom_idx in binding_atoms:
        neighbors = nl.get_neighbors(atom_idx)[0]
        surface_neighbors = [n for n in neighbors if atoms[n].tag >= 1]
        aoi.extend(surface_neighbors)
    
    aoi = sorted(set(aoi))
    
    edges = []
    for i in range(len(atoms)):
        neighbors, _ = nl.get_neighbors(i)
        for j in neighbors:
            edges.append([i, j])
    
    edges = np.array(edges)
    if len(edges) == 0:
        return None
    
    edge_dists = BFS(edges, ads_indices[0], len(atoms))
    
    gIds = ads_indices.copy()
    for t, n in [(0, 1), (1, 2), (2, 2), (3, 3)]:  
        for a in atoms:
            if (a.tag == t and 
                edge_dists[a.index] <= n and 
                a.index not in gIds):
                gIds.append(a.index)
    
    gIds = sorted(gIds)
    
    edges = edges[np.all(np.isin(edges, gIds), axis=1)]
    
    node_onehot = np.zeros((len(gIds), len(onehot_labels) + 2))
    for i, j in enumerate(gIds):
        symbol = atoms[j].symbol
        if symbol in onehot_labels:
            node_onehot[i, onehot_labels.index(symbol)] = 1
        node_onehot[i, -2] = atoms[j].tag
        if j in aoi:
            node_onehot[i, -1] = 1
    
    id_map = {g: i for i, g in enumerate(gIds)}
    edges = np.array([[id_map[e[0]], id_map[e[1]]] for e in edges])
    
    torch_edges = torch.tensor(np.transpose(edges), dtype=torch.long)
    torch_nodes = torch.tensor(node_onehot, dtype=torch.float)
    
    graph = Data(
        x=torch_nodes,
        edge_index=torch_edges,
        onehot_labels=onehot_labels
    )
    
    if data_file:
        with open(data_file, 'r') as f:
            energy_data = json.load(f)
        
        energy_key = f"energies_{adsorbate_type}_ads_raw"
        if energy_key in energy_data and len(energy_data[energy_key]) > site_index:
            energy = energy_data[energy_key][site_index]
            graph.y = energy
            graph.ads = adsorbate_type
    
    return graph

def process_data_directories(root_dir="C:/Users/Tseh/Documents/Files/HEA/hea_project"):
    source_dirs = {
        "triplets": os.path.join(root_dir, "train_data_dft/HEA_results_fives"),
    }
    
    output_dir = os.path.join(root_dir, "train_data_graphs/graphs")
    os.makedirs(output_dir, exist_ok=True)
    
    all_elements = set()
    print("Scanning elements..")
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
    
    for split_name, alloy_list in splits.items():
        print(f"\nProcessing {split_name} set ({len(alloy_list)} alloys)...")
        graph_list = []
        
        for category, alloy_dir, alloy_path in tqdm(alloy_list):
            xyz_file = os.path.join(alloy_path, "geometry.xyz")
            data_file = os.path.join(alloy_path, "data.json")
            
            for adsorbate_type in ['H', 'S']:
                for site_index in range(9):
                    try:
                        graph = create_site_specific_graph(
                            xyz_file, data_file, onehot_labels, adsorbate_type, site_index
                        )
                        
                        if graph is not None and hasattr(graph, 'y'):
                            graph_list.append(graph)
                    except Exception as e:
                        print(f"Error processing {alloy_path}, {adsorbate_type}, site {site_index}: {str(e)}")
        
        output_path = os.path.join(output_dir, f"{split_name}.graphs")
        with open(output_path, 'wb') as output:
            pickle.dump(graph_list, output)
        
        print(f"Saved {len(graph_list)} graphs to {output_path}")
    
    print("\nProcessing completed!")

if __name__ == "__main__":
    process_data_directories()