#!/usr/bin/env python3
"""
MOLECULAR DYNAMICS WITH NEIGHBOR LISTS
======================================

Advanced optimization using neighbor lists to reduce O(N²) to O(N)
scaling with spatial data structures and cell lists.
"""

import numpy as np
import numba as nb
from numba import njit, prange
import time
import warnings
warnings.filterwarnings("ignore")

# Pre-computed lookup tables for expensive functions
SQRT_TABLE_SIZE = 10000
MAX_DISTANCE_SQ = 25.0  # 5.0^2
sqrt_table = np.sqrt(np.linspace(0, MAX_DISTANCE_SQ, SQRT_TABLE_SIZE))
inv_sqrt_table = 1.0 / np.sqrt(np.linspace(0.01, MAX_DISTANCE_SQ, SQRT_TABLE_SIZE))

@njit(fastmath=True)
def fast_sqrt(x):
    """Fast square root using lookup table"""
    if x >= MAX_DISTANCE_SQ:
        return np.sqrt(x)
    
    idx = int(x * (SQRT_TABLE_SIZE - 1) / MAX_DISTANCE_SQ)
    if idx >= SQRT_TABLE_SIZE:
        idx = SQRT_TABLE_SIZE - 1
    
    return sqrt_table[idx]

@njit(fastmath=True)
def fast_inv_sqrt(x):
    """Fast inverse square root using lookup table"""
    if x >= MAX_DISTANCE_SQ or x < 0.01:
        return 1.0 / np.sqrt(x)
    
    idx = int((x - 0.01) * (SQRT_TABLE_SIZE - 1) / (MAX_DISTANCE_SQ - 0.01))
    if idx >= SQRT_TABLE_SIZE:
        idx = SQRT_TABLE_SIZE - 1
        
    return inv_sqrt_table[idx]

@njit(fastmath=True)
def build_cell_list(positions, box_size, cell_size):
    """Build cell list for spatial decomposition"""
    n_atoms = positions.shape[0]
    n_cells_1d = int(box_size / cell_size) + 1
    n_cells = n_cells_1d ** 3
    
    # Cell lists: cell_heads[cell] = first atom in cell, cell_list[atom] = next atom
    cell_heads = np.full(n_cells, -1, dtype=nb.int32)
    cell_list = np.full(n_atoms, -1, dtype=nb.int32)
    
    for i in range(n_atoms):
        # Find cell indices
        cx = int((positions[i, 0] + box_size/2) / cell_size)
        cy = int((positions[i, 1] + box_size/2) / cell_size) 
        cz = int((positions[i, 2] + box_size/2) / cell_size)
        
        # Clamp to valid range
        cx = max(0, min(cx, n_cells_1d - 1))
        cy = max(0, min(cy, n_cells_1d - 1))
        cz = max(0, min(cz, n_cells_1d - 1))
        
        cell = cx + cy * n_cells_1d + cz * n_cells_1d * n_cells_1d
        
        # Add atom to cell
        cell_list[i] = cell_heads[cell]
        cell_heads[cell] = i
    
    return cell_heads, cell_list, n_cells_1d

@njit(fastmath=True)
def build_neighbor_list(positions, cell_heads, cell_list, n_cells_1d, 
                       box_size, cell_size, cutoff_sq):
    """Build neighbor list using cell lists"""
    n_atoms = positions.shape[0]
    max_neighbors = 50  # Estimate
    
    neighbor_list = np.full((n_atoms, max_neighbors), -1, dtype=nb.int32)
    neighbor_count = np.zeros(n_atoms, dtype=nb.int32)
    
    for i in range(n_atoms):
        # Find cell of atom i
        cx = int((positions[i, 0] + box_size/2) / cell_size)
        cy = int((positions[i, 1] + box_size/2) / cell_size)
        cz = int((positions[i, 2] + box_size/2) / cell_size)
        
        # Clamp to valid range
        cx = max(0, min(cx, n_cells_1d - 1))
        cy = max(0, min(cy, n_cells_1d - 1))
        cz = max(0, min(cz, n_cells_1d - 1))
        
        # Search neighboring cells
        for dcx in range(-1, 2):
            for dcy in range(-1, 2):
                for dcz in range(-1, 2):
                    ncx = cx + dcx
                    ncy = cy + dcy
                    ncz = cz + dcz
                    
                    # Check bounds
                    if (ncx < 0 or ncx >= n_cells_1d or
                        ncy < 0 or ncy >= n_cells_1d or
                        ncz < 0 or ncz >= n_cells_1d):
                        continue
                    
                    cell = ncx + ncy * n_cells_1d + ncz * n_cells_1d * n_cells_1d
                    
                    # Iterate through atoms in this cell
                    j = cell_heads[cell]
                    while j != -1:
                        if j > i:  # Avoid double counting
                            # Check distance
                            dx = positions[j, 0] - positions[i, 0]
                            dy = positions[j, 1] - positions[i, 1]
                            dz = positions[j, 2] - positions[i, 2]
                            r2 = dx*dx + dy*dy + dz*dz
                            
                            if r2 < cutoff_sq:
                                if neighbor_count[i] < max_neighbors:
                                    neighbor_list[i, neighbor_count[i]] = j
                                    neighbor_count[i] += 1
                                if neighbor_count[j] < max_neighbors:
                                    neighbor_list[j, neighbor_count[j]] = i
                                    neighbor_count[j] += 1
                        
                        j = cell_list[j]
    
    return neighbor_list, neighbor_count

@njit(parallel=True, fastmath=True)
def compute_forces_neighbor_list(positions, neighbor_list, neighbor_count, forces):
    """Compute forces using neighbor lists"""
    n_atoms = positions.shape[0]
    forces.fill(0.0)
    
    # Lennard-Jones parameters
    sigma = 0.3165  # nm
    epsilon = 0.6502  # kJ/mol
    sigma6 = sigma**6
    sigma12 = sigma6 * sigma6
    
    for i in prange(n_atoms):
        for neighbor_idx in range(neighbor_count[i]):
            j = neighbor_list[i, neighbor_idx]
            if j == -1:
                break
            
            # Only compute once per pair (i < j guaranteed by neighbor list construction)
            if i < j:
                # Distance vector
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                r2 = dx*dx + dy*dy + dz*dz
                
                if r2 > 0.01:  # Avoid division by zero
                    r2_inv = 1.0 / r2
                    r6_inv = r2_inv * r2_inv * r2_inv
                    r12_inv = r6_inv * r6_inv
                    
                    # Lennard-Jones force
                    lj_force = 24.0 * epsilon * (2.0 * sigma12 * r12_inv - sigma6 * r6_inv) * r2_inv
                    
                    # Coulomb force (simplified)
                    r = fast_sqrt(r2)
                    coulomb_force = 0.1 / (r2 * r)
                    
                    total_force = lj_force + coulomb_force
                    
                    fx = total_force * dx
                    fy = total_force * dy
                    fz = total_force * dz
                    
                    # Newton's third law
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[i, 2] += fz
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                    forces[j, 2] -= fz

@njit(parallel=True, fastmath=True)
def compute_bonded_forces_optimized(positions, bonds, angles, forces):
    """Compute bonded forces with optimizations"""
    n_bonds = bonds.shape[0]
    n_angles = angles.shape[0]
    
    # Bond forces
    bond_k = 4000.0  # kJ/mol/nm²
    bond_r0 = 0.1    # nm
    
    for b in prange(n_bonds):
        i, j = bonds[b, 0], bonds[b, 1]
        
        dx = positions[j, 0] - positions[i, 0]
        dy = positions[j, 1] - positions[i, 1]
        dz = positions[j, 2] - positions[i, 2]
        
        r = fast_sqrt(dx*dx + dy*dy + dz*dz)
        
        if r > 1e-10:
            dr = r - bond_r0
            force_magnitude = -2.0 * bond_k * dr / r
            
            fx = force_magnitude * dx
            fy = force_magnitude * dy
            fz = force_magnitude * dz
            
            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[i, 2] += fz
            forces[j, 0] -= fx
            forces[j, 1] -= fy
            forces[j, 2] -= fz
    
    # Angle forces  
    angle_k = 400.0   # kJ/mol/rad²
    angle_theta0 = 1.9106  # radians (~109.5°)
    
    for a in prange(n_angles):
        i, j, k = angles[a, 0], angles[a, 1], angles[a, 2]
        
        # Vectors from central atom j
        dx1 = positions[i, 0] - positions[j, 0]
        dy1 = positions[i, 1] - positions[j, 1]
        dz1 = positions[i, 2] - positions[j, 2]
        
        dx2 = positions[k, 0] - positions[j, 0]
        dy2 = positions[k, 1] - positions[j, 1]
        dz2 = positions[k, 2] - positions[j, 2]
        
        r1 = fast_sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
        r2 = fast_sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
        
        if r1 > 1e-10 and r2 > 1e-10:
            cos_theta = (dx1*dx2 + dy1*dy2 + dz1*dz2) / (r1 * r2)
            cos_theta = max(-0.99999, min(0.99999, cos_theta))  # Clamp
            
            theta = np.arccos(cos_theta)
            dtheta = theta - angle_theta0
            
            if abs(np.sin(theta)) > 1e-6:
                force_const = -2.0 * angle_k * dtheta / np.sin(theta)
                
                # Force calculation (simplified)
                factor1 = force_const / (r1 * r2)
                factor2 = cos_theta / (r1 * r1)
                factor3 = cos_theta / (r2 * r2)
                
                fx1 = factor1 * dx2 - factor2 * dx1
                fy1 = factor1 * dy2 - factor2 * dy1
                fz1 = factor1 * dz2 - factor2 * dz1
                
                fx3 = factor1 * dx1 - factor3 * dx2
                fy3 = factor1 * dy1 - factor3 * dy2
                fz3 = factor1 * dz1 - factor3 * dz2
                
                forces[i, 0] += fx1
                forces[i, 1] += fy1
                forces[i, 2] += fz1
                forces[k, 0] += fx3
                forces[k, 1] += fy3
                forces[k, 2] += fz3
                forces[j, 0] -= fx1 + fx3
                forces[j, 1] -= fy1 + fy3
                forces[j, 2] -= fz1 + fz3

@njit(parallel=True, fastmath=True)
def integrate_leap_frog_optimized(positions, velocities, forces, masses, dt):
    """Optimized leap-frog integration"""
    n_atoms = positions.shape[0]
    
    for i in prange(n_atoms):
        inv_mass = 1.0 / masses[i]
        
        # Velocity update (half step)
        velocities[i, 0] += forces[i, 0] * inv_mass * dt * 0.5
        velocities[i, 1] += forces[i, 1] * inv_mass * dt * 0.5
        velocities[i, 2] += forces[i, 2] * inv_mass * dt * 0.5
        
        # Position update
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt
        positions[i, 2] += velocities[i, 2] * dt

def create_optimized_system(n_molecules=500):
    """Create optimized water system"""
    n_atoms = n_molecules * 3
    
    # Create positions in a rough cubic arrangement
    cube_size = int(np.ceil(n_molecules**(1/3)))
    spacing = 0.4
    
    positions = np.zeros((n_atoms, 3))
    velocities = np.random.normal(0, 0.1, (n_atoms, 3))
    masses = np.ones(n_atoms)
    
    # Oxygen masses
    masses[::3] = 15.999
    # Hydrogen masses
    masses[1::3] = 1.008
    masses[2::3] = 1.008
    
    atom_idx = 0
    for i in range(cube_size):
        for j in range(cube_size):
            for k in range(cube_size):
                if atom_idx >= n_molecules:
                    break
                
                # Water molecule positions (O-H-H)
                base_pos = np.array([i * spacing, j * spacing, k * spacing])
                
                # Oxygen
                positions[atom_idx * 3] = base_pos
                # Hydrogen 1
                positions[atom_idx * 3 + 1] = base_pos + [0.1, 0.0, 0.0]
                # Hydrogen 2  
                positions[atom_idx * 3 + 2] = base_pos + [-0.033, 0.094, 0.0]
                
                atom_idx += 1
    
    # Create bonds (O-H)
    bonds = []
    for mol in range(n_molecules):
        base = mol * 3
        bonds.append([base, base + 1])      # O-H1
        bonds.append([base, base + 2])      # O-H2
    bonds = np.array(bonds, dtype=np.int32)
    
    # Create angles (H-O-H)
    angles = []
    for mol in range(n_molecules):
        base = mol * 3
        angles.append([base + 1, base, base + 2])  # H1-O-H2
    angles = np.array(angles, dtype=np.int32)
    
    return positions, velocities, masses, bonds, angles

def run_neighbor_list_simulation(n_molecules=500, n_steps=100, dt=0.001):
    """Run optimized simulation with neighbor lists"""
    print(f"🚀 NEIGHBOR LIST SIMULATION")
    print(f"Molecules: {n_molecules}, Steps: {n_steps}")
    
    # Create system
    setup_start = time.perf_counter()
    positions, velocities, masses, bonds, angles = create_optimized_system(n_molecules)
    n_atoms = positions.shape[0]
    forces = np.zeros((n_atoms, 3))
    setup_time = time.perf_counter() - setup_start
    
    # Neighbor list parameters
    box_size = 10.0
    cutoff = 1.0
    cutoff_sq = cutoff * cutoff
    cell_size = cutoff / 2.0  # Smaller cells for better distribution
    
    print(f"Setup time: {setup_time:.4f}s")
    print(f"Box size: {box_size:.1f} nm")
    print(f"Cutoff: {cutoff:.1f} nm") 
    print(f"Cell size: {cell_size:.2f} nm")
    
    # Initial neighbor list
    neighbor_start = time.perf_counter()
    cell_heads, cell_list, n_cells_1d = build_cell_list(positions, box_size, cell_size)
    neighbor_list, neighbor_count = build_neighbor_list(
        positions, cell_heads, cell_list, n_cells_1d, box_size, cell_size, cutoff_sq
    )
    neighbor_time = time.perf_counter() - neighbor_start
    
    total_neighbors = np.sum(neighbor_count)
    print(f"Neighbor list build: {neighbor_time:.4f}s")
    print(f"Total neighbors: {total_neighbors}")
    print(f"Avg neighbors per atom: {total_neighbors/n_atoms:.1f}")
    
    # Simulation loop
    sim_start = time.perf_counter()
    rebuild_frequency = 10  # Rebuild every N steps
    
    for step in range(n_steps):
        # Rebuild neighbor list periodically
        if step % rebuild_frequency == 0 and step > 0:
            cell_heads, cell_list, n_cells_1d = build_cell_list(positions, box_size, cell_size)
            neighbor_list, neighbor_count = build_neighbor_list(
                positions, cell_heads, cell_list, n_cells_1d, box_size, cell_size, cutoff_sq
            )
        
        # Compute forces
        compute_forces_neighbor_list(positions, neighbor_list, neighbor_count, forces)
        compute_bonded_forces_optimized(positions, bonds, angles, forces)
        
        # Integration
        integrate_leap_frog_optimized(positions, velocities, forces, masses, dt)
    
    sim_time = time.perf_counter() - sim_start
    total_time = setup_time + neighbor_time + sim_time
    
    print(f"Simulation time: {sim_time:.4f}s")
    print(f"Total time: {total_time:.4f}s")
    print(f"Time per step: {sim_time/n_steps*1000:.2f} ms")
    
    # Calculate final properties
    total_energy = np.sum(0.5 * masses[:, np.newaxis] * velocities**2)
    rms_velocity = np.sqrt(np.mean(velocities**2))
    
    print(f"Final kinetic energy: {total_energy:.2f}")
    print(f"RMS velocity: {rms_velocity:.4f}")
    
    return total_time

if __name__ == "__main__":
    print("🔬 MOLECULAR DYNAMICS WITH ADVANCED OPTIMIZATIONS")
    print("=" * 60)
    
    # Test different system sizes
    sizes = [100, 250, 500]
    
    for n_mol in sizes:
        print(f"\n📊 Testing {n_mol} molecules:")
        print("-" * 40)
        time_taken = run_neighbor_list_simulation(n_mol, 50, 0.001)
        print(f"✅ Completed in {time_taken:.4f}s")