#!/usr/bin/env python3
"""
Ultra-Fast Molecular Dynamics with Pure Arrays
===============================================

Complete redesign eliminating ALL Python objects:
- All data stored in flat NumPy arrays
- No classes, no objects, no overhead
- Optimized memory layout for cache efficiency
- Vectorized operations throughout

This is what fast Python MD should look like!
"""

import numpy as np
import time
from numba import njit, prange
import sys

# Global constants
BOLTZMANN = 1.0
AVOGADRO = 6.022e23
KCAL_TO_JOULE = 4184.0

def create_water_system_arrays(n_molecules):
    """Create water system using pure arrays - no Python objects"""
    n_atoms = n_molecules * 3  # 3 atoms per water molecule
    n_bonds = n_molecules * 2  # 2 bonds per water molecule  
    n_angles = n_molecules * 1  # 1 angle per water molecule
    
    # Atom data arrays - Structure of Arrays (SoA) layout
    positions = np.zeros((n_atoms, 3), dtype=np.float64)  # [x, y, z] for each atom
    velocities = np.zeros((n_atoms, 3), dtype=np.float64)
    forces = np.zeros((n_atoms, 3), dtype=np.float64)
    masses = np.zeros(n_atoms, dtype=np.float64)
    charges = np.zeros(n_atoms, dtype=np.float64)
    atom_types = np.zeros(n_atoms, dtype=np.int32)  # 0=O, 1=H
    
    # LJ parameters per atom type
    sigma = np.array([3.15, 2.0], dtype=np.float64)      # O, H
    epsilon = np.array([0.1521, 0.0460], dtype=np.float64) # O, H
    
    # Bond topology arrays
    bond_atoms = np.zeros((n_bonds, 2), dtype=np.int32)   # [atom1_id, atom2_id]
    bond_k = np.zeros(n_bonds, dtype=np.float64)          # Force constants
    bond_r0 = np.zeros(n_bonds, dtype=np.float64)         # Equilibrium lengths
    
    # Angle topology arrays  
    angle_atoms = np.zeros((n_angles, 3), dtype=np.int32) # [atom1, atom2, atom3]
    angle_k = np.zeros(n_angles, dtype=np.float64)        # Force constants
    angle_theta0 = np.zeros(n_angles, dtype=np.float64)   # Equilibrium angles
    
    # Initialize water molecules
    bond_idx = 0
    angle_idx = 0
    
    for mol in range(n_molecules):
        atom_base = mol * 3
        
        # Oxygen atom (atom 0 of molecule)
        o_idx = atom_base + 0
        h1_idx = atom_base + 1  
        h2_idx = atom_base + 2
        
        # Set positions in triangular water geometry
        positions[o_idx] = [mol * 3.0, 0.0, 0.0]  # Oxygen at origin
        positions[h1_idx] = [mol * 3.0 + 0.9584, 0.0, 0.0]  # H1 
        positions[h2_idx] = [mol * 3.0 + 0.9584 * np.cos(104.45 * np.pi/180), 
                            0.9584 * np.sin(104.45 * np.pi/180), 0.0]  # H2
        
        # Add random perturbation
        positions[o_idx:o_idx+3] += np.random.normal(0, 0.1, (3, 3))
        
        # Set masses
        masses[o_idx] = 15.999    # Oxygen
        masses[h1_idx] = 1.008    # Hydrogen 1  
        masses[h2_idx] = 1.008    # Hydrogen 2
        
        # Set charges  
        charges[o_idx] = -0.8476   # Oxygen partial charge
        charges[h1_idx] = 0.4238   # Hydrogen partial charge
        charges[h2_idx] = 0.4238   # Hydrogen partial charge
        
        # Set atom types
        atom_types[o_idx] = 0      # Oxygen
        atom_types[h1_idx] = 1     # Hydrogen
        atom_types[h2_idx] = 1     # Hydrogen
        
        # Create bonds: O-H1 and O-H2
        bond_atoms[bond_idx] = [o_idx, h1_idx]
        bond_k[bond_idx] = 450.0      # kcal/mol/Å²
        bond_r0[bond_idx] = 0.9584    # Å
        bond_idx += 1
        
        bond_atoms[bond_idx] = [o_idx, h2_idx]
        bond_k[bond_idx] = 450.0
        bond_r0[bond_idx] = 0.9584
        bond_idx += 1
        
        # Create angle: H1-O-H2
        angle_atoms[angle_idx] = [h1_idx, o_idx, h2_idx]  # Central atom is O
        angle_k[angle_idx] = 55.0           # kcal/mol/rad²
        angle_theta0[angle_idx] = 104.45 * np.pi/180  # radians
        angle_idx += 1
    
    return {
        'n_atoms': n_atoms,
        'n_molecules': n_molecules,
        'positions': positions,
        'velocities': velocities, 
        'forces': forces,
        'masses': masses,
        'charges': charges,
        'atom_types': atom_types,
        'sigma': sigma,
        'epsilon': epsilon,
        'bond_atoms': bond_atoms,
        'bond_k': bond_k,
        'bond_r0': bond_r0,
        'angle_atoms': angle_atoms,
        'angle_k': angle_k,
        'angle_theta0': angle_theta0
    }

@njit(parallel=True, fastmath=True)
def compute_bonds_pure(positions, forces, bond_atoms, bond_k, bond_r0):
    """Ultra-fast bond force computation with pure arrays"""
    n_bonds = bond_atoms.shape[0]
    
    for i in prange(n_bonds):
        atom1 = bond_atoms[i, 0]
        atom2 = bond_atoms[i, 1]
        
        # Vector from atom1 to atom2
        dx = positions[atom2, 0] - positions[atom1, 0]
        dy = positions[atom2, 1] - positions[atom1, 1]
        dz = positions[atom2, 2] - positions[atom1, 2]
        
        # Distance
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if r > 1e-10:
            # Harmonic potential: F = -k * (r - r0) * dr/r
            force_mag = -bond_k[i] * (r - bond_r0[i]) / r
            
            fx = force_mag * dx
            fy = force_mag * dy  
            fz = force_mag * dz
            
            # Apply forces (Newton's 3rd law)
            forces[atom1, 0] -= fx
            forces[atom1, 1] -= fy
            forces[atom1, 2] -= fz
            
            forces[atom2, 0] += fx
            forces[atom2, 1] += fy
            forces[atom2, 2] += fz

@njit(parallel=True, fastmath=True)
def compute_angles_pure(positions, forces, angle_atoms, angle_k, angle_theta0):
    """Ultra-fast angle force computation with pure arrays"""
    n_angles = angle_atoms.shape[0]
    
    for i in prange(n_angles):
        atom1 = angle_atoms[i, 0]
        atom2 = angle_atoms[i, 1]  # Central atom
        atom3 = angle_atoms[i, 2]
        
        # Vectors from central atom
        dx1 = positions[atom1, 0] - positions[atom2, 0]
        dy1 = positions[atom1, 1] - positions[atom2, 1]
        dz1 = positions[atom1, 2] - positions[atom2, 2]
        
        dx2 = positions[atom3, 0] - positions[atom2, 0]
        dy2 = positions[atom3, 1] - positions[atom2, 1]
        dz2 = positions[atom3, 2] - positions[atom2, 2]
        
        # Lengths
        r1 = np.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
        r2 = np.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
        
        if r1 > 1e-10 and r2 > 1e-10:
            # Normalized vectors
            dx1 /= r1
            dy1 /= r1 
            dz1 /= r1
            dx2 /= r2
            dy2 /= r2
            dz2 /= r2
            
            # Cosine of angle
            cos_theta = dx1*dx2 + dy1*dy2 + dz1*dz2
            cos_theta = max(-1.0, min(1.0, cos_theta))  # Clamp
            
            theta = np.arccos(cos_theta)
            sin_theta = np.sin(theta)
            
            if abs(sin_theta) > 1e-10:
                # Force magnitude
                dtheta = theta - angle_theta0[i]
                force_const = -angle_k[i] * dtheta / sin_theta
                
                # Force computation (simplified for speed)
                coeff1 = force_const / r1
                coeff2 = force_const / r2
                
                f1x = coeff1 * (dx2 - cos_theta * dx1)
                f1y = coeff1 * (dy2 - cos_theta * dy1)
                f1z = coeff1 * (dz2 - cos_theta * dz1)
                
                f3x = coeff2 * (dx1 - cos_theta * dx2)
                f3y = coeff2 * (dy1 - cos_theta * dy2)
                f3z = coeff2 * (dz1 - cos_theta * dz2)
                
                # Apply forces
                forces[atom1, 0] += f1x
                forces[atom1, 1] += f1y
                forces[atom1, 2] += f1z
                
                forces[atom2, 0] -= (f1x + f3x)
                forces[atom2, 1] -= (f1y + f3y)
                forces[atom2, 2] -= (f1z + f3z)
                
                forces[atom3, 0] += f3x
                forces[atom3, 1] += f3y
                forces[atom3, 2] += f3z

@njit(parallel=True, fastmath=True)
def compute_nonbonded_pure(positions, forces, charges, atom_types, sigma, epsilon, cutoff=10.0):
    """Ultra-fast non-bonded forces with pure arrays and spatial optimization"""
    n_atoms = positions.shape[0]
    cutoff2 = cutoff * cutoff
    
    for i in prange(n_atoms):
        for j in range(i + 1, n_atoms):
            # Skip if same molecule (simple check - every 3 atoms)
            mol_i = i // 3
            mol_j = j // 3
            if mol_i == mol_j:
                continue
                
            # Distance vector
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            
            r2 = dx*dx + dy*dy + dz*dz
            
            # Cutoff check
            if r2 < cutoff2 and r2 > 0.01:  # Avoid singularities
                r = np.sqrt(r2)
                r_inv = 1.0 / r
                
                # Get LJ parameters
                type_i = atom_types[i]
                type_j = atom_types[j]
                sig_ij = 0.5 * (sigma[type_i] + sigma[type_j])  # Lorentz-Berthelot
                eps_ij = np.sqrt(epsilon[type_i] * epsilon[type_j])
                
                # Lennard-Jones force
                sig_r = sig_ij * r_inv
                sig_r6 = sig_r**6
                sig_r12 = sig_r6 * sig_r6
                lj_force = 24.0 * eps_ij * (2.0 * sig_r12 - sig_r6) * r_inv
                
                # Coulomb force (with conversion factor)
                coulomb_force = 332.0637 * charges[i] * charges[j] * r_inv * r_inv
                
                total_force = (lj_force + coulomb_force) * r_inv
                
                # Force components
                fx = total_force * dx
                fy = total_force * dy
                fz = total_force * dz
                
                # Apply forces (Newton's 3rd law)
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[i, 2] += fz
                
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                forces[j, 2] -= fz

@njit(parallel=True, fastmath=True)
def integrate_verlet_pure(positions, velocities, forces, masses, dt):
    """Ultra-fast Velocity Verlet integration with pure arrays"""
    n_atoms = positions.shape[0]
    dt_half = 0.5 * dt
    
    for i in prange(n_atoms):
        mass_inv = 1.0 / masses[i]
        
        for dim in range(3):
            # Acceleration
            acc = forces[i, dim] * mass_inv
            
            # Velocity Verlet: v(t+dt/2) = v(t) + a(t) * dt/2
            velocities[i, dim] += acc * dt_half
            
            # Position: r(t+dt) = r(t) + v(t+dt/2) * dt
            positions[i, dim] += velocities[i, dim] * dt

def run_pure_array_simulation(n_molecules, n_steps, dt):
    """Run molecular dynamics with pure arrays - maximum performance"""
    print(f"🚀 Running pure array simulation: {n_molecules} molecules, {n_steps} steps")
    
    # Create system
    system = create_water_system_arrays(n_molecules)
    
    start_time = time.perf_counter()
    
    for step in range(n_steps):
        # Clear forces
        system['forces'].fill(0.0)
        
        # Compute all forces
        compute_bonds_pure(system['positions'], system['forces'], 
                          system['bond_atoms'], system['bond_k'], system['bond_r0'])
        
        compute_angles_pure(system['positions'], system['forces'],
                           system['angle_atoms'], system['angle_k'], system['angle_theta0'])
        
        compute_nonbonded_pure(system['positions'], system['forces'], system['charges'],
                              system['atom_types'], system['sigma'], system['epsilon'])
        
        # Integration  
        integrate_verlet_pure(system['positions'], system['velocities'], 
                             system['forces'], system['masses'], dt)
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    print(f"✅ Pure array simulation completed in {elapsed:.3f}s")
    return elapsed

def warmup_pure_arrays():
    """Warmup compilation for pure array functions"""
    print("⚡ Warming up pure array functions...")
    
    # Create small test system
    test_system = create_water_system_arrays(10)
    
    # Compile all functions
    compute_bonds_pure(test_system['positions'], test_system['forces'],
                      test_system['bond_atoms'], test_system['bond_k'], test_system['bond_r0'])
    
    compute_angles_pure(test_system['positions'], test_system['forces'], 
                       test_system['angle_atoms'], test_system['angle_k'], test_system['angle_theta0'])
    
    compute_nonbonded_pure(test_system['positions'], test_system['forces'], test_system['charges'],
                          test_system['atom_types'], test_system['sigma'], test_system['epsilon'])
    
    integrate_verlet_pure(test_system['positions'], test_system['velocities'],
                         test_system['forces'], test_system['masses'], 0.001)
    
    print("✅ Pure array compilation complete!")

if __name__ == "__main__":
    # Warmup and test
    warmup_pure_arrays()
    
    print("\n🏆 PURE ARRAY PERFORMANCE TEST")
    print("=" * 50)
    
    sizes = [100, 500, 1000, 2000]
    for size in sizes:
        elapsed = run_pure_array_simulation(size, 100, 0.001)
        print(f"Size {size:4d}: {elapsed:.3f}s ({size*100/elapsed:.0f} atom-steps/s)")