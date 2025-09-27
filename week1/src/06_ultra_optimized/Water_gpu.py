#!/usr/bin/env python3
"""
GPU-Accelerated Molecular Dynamics with CuPy
==============================================

Massively parallel molecular dynamics using GPU:
- All computations on GPU with CuPy
- Thousands of parallel threads
- Minimal CPU-GPU data transfer  
- Custom CUDA kernels for maximum performance

This is GPU-accelerated Python MD!
"""

import numpy as np
import time
import sys

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🔥 GPU detected - CuPy available!")
except ImportError:
    print("❌ CuPy not installed - falling back to CPU")
    import numpy as cp  # Fallback to numpy
    GPU_AVAILABLE = False

def create_gpu_water_system(n_molecules):
    """Create water system on GPU using CuPy arrays"""
    n_atoms = n_molecules * 3
    n_bonds = n_molecules * 2  
    n_angles = n_molecules * 1
    
    # All arrays created directly on GPU
    positions = cp.zeros((n_atoms, 3), dtype=cp.float32)  # Use float32 for GPU speed
    velocities = cp.zeros((n_atoms, 3), dtype=cp.float32)
    forces = cp.zeros((n_atoms, 3), dtype=cp.float32)
    masses = cp.zeros(n_atoms, dtype=cp.float32)
    charges = cp.zeros(n_atoms, dtype=cp.float32)
    atom_types = cp.zeros(n_atoms, dtype=cp.int32)
    
    # LJ parameters
    sigma = cp.array([3.15, 2.0], dtype=cp.float32)
    epsilon = cp.array([0.1521, 0.0460], dtype=cp.float32)
    
    # Topology arrays on GPU
    bond_atoms = cp.zeros((n_bonds, 2), dtype=cp.int32)
    bond_k = cp.zeros(n_bonds, dtype=cp.float32)
    bond_r0 = cp.zeros(n_bonds, dtype=cp.float32)
    
    angle_atoms = cp.zeros((n_angles, 3), dtype=cp.int32)
    angle_k = cp.zeros(n_angles, dtype=cp.float32)
    angle_theta0 = cp.zeros(n_angles, dtype=cp.float32)
    
    # Initialize on CPU then transfer (more efficient for setup)
    pos_cpu = np.zeros((n_atoms, 3), dtype=np.float32)
    masses_cpu = np.zeros(n_atoms, dtype=np.float32)
    charges_cpu = np.zeros(n_atoms, dtype=np.float32)
    atom_types_cpu = np.zeros(n_atoms, dtype=np.int32)
    bond_atoms_cpu = np.zeros((n_bonds, 2), dtype=np.int32)
    bond_k_cpu = np.zeros(n_bonds, dtype=np.float32)
    bond_r0_cpu = np.zeros(n_bonds, dtype=np.float32)
    angle_atoms_cpu = np.zeros((n_angles, 3), dtype=np.int32)
    angle_k_cpu = np.zeros(n_angles, dtype=np.float32)
    angle_theta0_cpu = np.zeros(n_angles, dtype=np.float32)
    
    bond_idx = 0
    angle_idx = 0
    
    for mol in range(n_molecules):
        atom_base = mol * 3
        o_idx = atom_base + 0
        h1_idx = atom_base + 1  
        h2_idx = atom_base + 2
        
        # Water geometry
        pos_cpu[o_idx] = [mol * 3.0, 0.0, 0.0]
        pos_cpu[h1_idx] = [mol * 3.0 + 0.9584, 0.0, 0.0]
        pos_cpu[h2_idx] = [mol * 3.0 + 0.9584 * np.cos(104.45 * np.pi/180), 
                          0.9584 * np.sin(104.45 * np.pi/180), 0.0]
        
        # Random perturbation
        pos_cpu[o_idx:o_idx+3] += np.random.normal(0, 0.1, (3, 3)).astype(np.float32)
        
        # Set properties
        masses_cpu[o_idx] = 15.999
        masses_cpu[h1_idx] = 1.008
        masses_cpu[h2_idx] = 1.008
        
        charges_cpu[o_idx] = -0.8476
        charges_cpu[h1_idx] = 0.4238
        charges_cpu[h2_idx] = 0.4238
        
        atom_types_cpu[o_idx] = 0
        atom_types_cpu[h1_idx] = 1
        atom_types_cpu[h2_idx] = 1
        
        # Bonds
        bond_atoms_cpu[bond_idx] = [o_idx, h1_idx]
        bond_k_cpu[bond_idx] = 450.0
        bond_r0_cpu[bond_idx] = 0.9584
        bond_idx += 1
        
        bond_atoms_cpu[bond_idx] = [o_idx, h2_idx]
        bond_k_cpu[bond_idx] = 450.0
        bond_r0_cpu[bond_idx] = 0.9584
        bond_idx += 1
        
        # Angles
        angle_atoms_cpu[angle_idx] = [h1_idx, o_idx, h2_idx]
        angle_k_cpu[angle_idx] = 55.0
        angle_theta0_cpu[angle_idx] = 104.45 * np.pi/180
        angle_idx += 1
    
    # Transfer to GPU
    positions[:] = cp.asarray(pos_cpu)
    masses[:] = cp.asarray(masses_cpu)
    charges[:] = cp.asarray(charges_cpu)
    atom_types[:] = cp.asarray(atom_types_cpu)
    bond_atoms[:] = cp.asarray(bond_atoms_cpu)
    bond_k[:] = cp.asarray(bond_k_cpu)
    bond_r0[:] = cp.asarray(bond_r0_cpu)
    angle_atoms[:] = cp.asarray(angle_atoms_cpu)
    angle_k[:] = cp.asarray(angle_k_cpu)
    angle_theta0[:] = cp.asarray(angle_theta0_cpu)
    
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

# Custom CUDA kernel for bond forces (if CuPy available)
bond_force_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_bond_forces(float* positions, float* forces, int* bond_atoms, 
                        float* bond_k, float* bond_r0, int n_bonds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bonds) return;
    
    int atom1 = bond_atoms[idx * 2];
    int atom2 = bond_atoms[idx * 2 + 1];
    
    float dx = positions[atom2 * 3] - positions[atom1 * 3];
    float dy = positions[atom2 * 3 + 1] - positions[atom1 * 3 + 1];
    float dz = positions[atom2 * 3 + 2] - positions[atom1 * 3 + 2];
    
    float r = sqrtf(dx*dx + dy*dy + dz*dz);
    
    if (r > 1e-10f) {
        float force_mag = -bond_k[idx] * (r - bond_r0[idx]) / r;
        
        float fx = force_mag * dx;
        float fy = force_mag * dy;
        float fz = force_mag * dz;
        
        atomicAdd(&forces[atom1 * 3], -fx);
        atomicAdd(&forces[atom1 * 3 + 1], -fy);
        atomicAdd(&forces[atom1 * 3 + 2], -fz);
        
        atomicAdd(&forces[atom2 * 3], fx);
        atomicAdd(&forces[atom2 * 3 + 1], fy);
        atomicAdd(&forces[atom2 * 3 + 2], fz);
    }
}
''', 'compute_bond_forces') if GPU_AVAILABLE else None

# Custom CUDA kernel for non-bonded forces
nonbonded_force_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_nonbonded_forces(float* positions, float* forces, float* charges,
                             int* atom_types, float* sigma, float* epsilon,
                             int n_atoms, float cutoff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;
    
    float cutoff2 = cutoff * cutoff;
    float xi = positions[i * 3];
    float yi = positions[i * 3 + 1]; 
    float zi = positions[i * 3 + 2];
    int type_i = atom_types[i];
    float qi = charges[i];
    
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    
    for (int j = i + 1; j < n_atoms; j++) {
        // Skip same molecule
        if ((i / 3) == (j / 3)) continue;
        
        float dx = positions[j * 3] - xi;
        float dy = positions[j * 3 + 1] - yi;
        float dz = positions[j * 3 + 2] - zi;
        
        float r2 = dx*dx + dy*dy + dz*dz;
        
        if (r2 < cutoff2 && r2 > 0.01f) {
            float r = sqrtf(r2);
            float r_inv = 1.0f / r;
            
            int type_j = atom_types[j];
            float sig_ij = 0.5f * (sigma[type_i] + sigma[type_j]);
            float eps_ij = sqrtf(epsilon[type_i] * epsilon[type_j]);
            
            // Lennard-Jones
            float sig_r = sig_ij * r_inv;
            float sig_r6 = sig_r * sig_r * sig_r * sig_r * sig_r * sig_r;
            float sig_r12 = sig_r6 * sig_r6;
            float lj_force = 24.0f * eps_ij * (2.0f * sig_r12 - sig_r6) * r_inv;
            
            // Coulomb
            float qj = charges[j];
            float coulomb_force = 332.0637f * qi * qj * r_inv * r_inv;
            
            float total_force = (lj_force + coulomb_force) * r_inv;
            
            fx += total_force * dx;
            fy += total_force * dy;
            fz += total_force * dz;
            
            // Newton's 3rd law for j
            atomicAdd(&forces[j * 3], -total_force * dx);
            atomicAdd(&forces[j * 3 + 1], -total_force * dy);
            atomicAdd(&forces[j * 3 + 2], -total_force * dz);
        }
    }
    
    atomicAdd(&forces[i * 3], fx);
    atomicAdd(&forces[i * 3 + 1], fy);
    atomicAdd(&forces[i * 3 + 2], fz);
}
''', 'compute_nonbonded_forces') if GPU_AVAILABLE else None

def compute_bonds_gpu(system):
    """GPU-accelerated bond force computation"""
    if not GPU_AVAILABLE:
        # Fallback to CPU
        n_bonds = system['bond_atoms'].shape[0]
        for i in range(n_bonds):
            atom1, atom2 = system['bond_atoms'][i]
            dx = system['positions'][atom2, 0] - system['positions'][atom1, 0]
            dy = system['positions'][atom2, 1] - system['positions'][atom1, 1]
            dz = system['positions'][atom2, 2] - system['positions'][atom1, 2]
            r = cp.sqrt(dx*dx + dy*dy + dz*dz)
            if r > 1e-10:
                force_mag = -system['bond_k'][i] * (r - system['bond_r0'][i]) / r
                fx = force_mag * dx
                fy = force_mag * dy  
                fz = force_mag * dz
                system['forces'][atom1, 0] -= fx
                system['forces'][atom1, 1] -= fy
                system['forces'][atom1, 2] -= fz
                system['forces'][atom2, 0] += fx
                system['forces'][atom2, 1] += fy
                system['forces'][atom2, 2] += fz
        return
    
    n_bonds = system['bond_atoms'].shape[0]
    if n_bonds == 0:
        return
        
    # Launch CUDA kernel
    block_size = 256
    grid_size = (n_bonds + block_size - 1) // block_size
    
    bond_force_kernel((grid_size,), (block_size,), (
        system['positions'].data.ptr, system['forces'].data.ptr,
        system['bond_atoms'].data.ptr, system['bond_k'].data.ptr,
        system['bond_r0'].data.ptr, n_bonds
    ))

def compute_nonbonded_gpu(system, cutoff=10.0):
    """GPU-accelerated non-bonded force computation"""
    n_atoms = system['n_atoms']
    
    if not GPU_AVAILABLE:
        # Fallback to CPU
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                mol_i = i // 3
                mol_j = j // 3
                if mol_i == mol_j:
                    continue
                    
                dx = system['positions'][j, 0] - system['positions'][i, 0]
                dy = system['positions'][j, 1] - system['positions'][i, 1]
                dz = system['positions'][j, 2] - system['positions'][i, 2]
                r2 = dx*dx + dy*dy + dz*dz
                
                if r2 < cutoff*cutoff and r2 > 0.01:
                    r = cp.sqrt(r2)
                    r_inv = 1.0 / r
                    
                    type_i = system['atom_types'][i]
                    type_j = system['atom_types'][j]
                    sig_ij = 0.5 * (system['sigma'][type_i] + system['sigma'][type_j])
                    eps_ij = cp.sqrt(system['epsilon'][type_i] * system['epsilon'][type_j])
                    
                    sig_r = sig_ij * r_inv
                    sig_r6 = sig_r**6
                    sig_r12 = sig_r6 * sig_r6
                    lj_force = 24.0 * eps_ij * (2.0 * sig_r12 - sig_r6) * r_inv
                    
                    coulomb_force = 332.0637 * system['charges'][i] * system['charges'][j] * r_inv * r_inv
                    
                    total_force = (lj_force + coulomb_force) * r_inv
                    
                    fx = total_force * dx
                    fy = total_force * dy
                    fz = total_force * dz
                    
                    system['forces'][i, 0] += fx
                    system['forces'][i, 1] += fy
                    system['forces'][i, 2] += fz
                    system['forces'][j, 0] -= fx
                    system['forces'][j, 1] -= fy
                    system['forces'][j, 2] -= fz
        return
    
    # Launch CUDA kernel
    block_size = 256
    grid_size = (n_atoms + block_size - 1) // block_size
    
    nonbonded_force_kernel((grid_size,), (block_size,), (
        system['positions'].data.ptr, system['forces'].data.ptr,
        system['charges'].data.ptr, system['atom_types'].data.ptr,
        system['sigma'].data.ptr, system['epsilon'].data.ptr,
        n_atoms, cp.float32(cutoff)
    ))

def integrate_gpu(system, dt):
    """GPU-accelerated Velocity Verlet integration"""
    dt_half = 0.5 * dt
    
    # Vectorized operations on GPU
    acc = system['forces'] / system['masses'][:, None]  # Broadcasting
    system['velocities'] += acc * dt_half
    system['positions'] += system['velocities'] * dt

def run_gpu_simulation(n_molecules, n_steps, dt):
    """Run GPU-accelerated molecular dynamics simulation"""
    print(f"🔥 Running GPU simulation: {n_molecules} molecules, {n_steps} steps")
    
    # Create system on GPU
    system = create_gpu_water_system(n_molecules)
    
    if GPU_AVAILABLE:
        # Synchronize GPU before timing
        cp.cuda.Stream.null.synchronize()
    
    start_time = time.perf_counter()
    
    for step in range(n_steps):
        # Clear forces
        system['forces'].fill(0.0)
        
        # Compute forces on GPU
        compute_bonds_gpu(system)
        compute_nonbonded_gpu(system)
        
        # Integration on GPU
        integrate_gpu(system, dt)
        
        # Optional: synchronize every few steps for stability
        if GPU_AVAILABLE and step % 10 == 0:
            cp.cuda.Stream.null.synchronize()
    
    if GPU_AVAILABLE:
        # Final synchronization for accurate timing
        cp.cuda.Stream.null.synchronize()
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    print(f"✅ GPU simulation completed in {elapsed:.3f}s")
    return elapsed

if __name__ == "__main__":
    if GPU_AVAILABLE:
        print(f"🔥 GPU Info: {cp.cuda.Device()}")
        print(f"🔥 GPU Memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB total")
    
    print("\n🏆 GPU PERFORMANCE TEST")
    print("=" * 50)
    
    sizes = [100, 500, 1000, 2000]
    for size in sizes:
        elapsed = run_gpu_simulation(size, 100, 0.001)
        throughput = size * 100 / elapsed
        print(f"Size {size:4d}: {elapsed:.3f}s ({throughput:.0f} molecule-steps/s)")