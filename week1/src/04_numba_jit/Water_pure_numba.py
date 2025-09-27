#!/usr/bin/env python3
import numpy as np
import time
import sys
from numba import njit, prange

# Pure NumPy/Numba implementation - no Python objects
@njit(parallel=True)
def numba_bond_forces(positions, forces, bonds, K_bond, L0_bond):
    """Pure NumPy bond force calculation with Numba"""
    n_bonds = bonds.shape[0]
    
    for i in prange(n_bonds):
        atom1_idx = bonds[i, 0]
        atom2_idx = bonds[i, 1]
        K = K_bond[i]
        L0 = L0_bond[i]
        
        # Vector from atom1 to atom2
        dx = positions[atom2_idx, 0] - positions[atom1_idx, 0]
        dy = positions[atom2_idx, 1] - positions[atom1_idx, 1] 
        dz = positions[atom2_idx, 2] - positions[atom1_idx, 2]
        
        # Distance
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if r > 0:
            # Force magnitude
            force_mag = K * (r - L0) / r
            
            # Force components
            fx = force_mag * dx
            fy = force_mag * dy
            fz = force_mag * dz
            
            # Apply forces (Newton's 3rd law)
            forces[atom1_idx, 0] += fx
            forces[atom1_idx, 1] += fy
            forces[atom1_idx, 2] += fz
            
            forces[atom2_idx, 0] -= fx
            forces[atom2_idx, 1] -= fy
            forces[atom2_idx, 2] -= fz

@njit(parallel=True)
def numba_angle_forces(positions, forces, angles, K_angle, theta0_angle):
    """Pure NumPy angle force calculation with Numba"""
    n_angles = angles.shape[0]
    
    for i in prange(n_angles):
        atom1_idx = angles[i, 0]
        atom2_idx = angles[i, 1]  # central atom
        atom3_idx = angles[i, 2]
        K = K_angle[i]
        theta0 = theta0_angle[i]
        
        # Vectors from central atom to others
        dx1 = positions[atom1_idx, 0] - positions[atom2_idx, 0]
        dy1 = positions[atom1_idx, 1] - positions[atom2_idx, 1]
        dz1 = positions[atom1_idx, 2] - positions[atom2_idx, 2]
        
        dx2 = positions[atom3_idx, 0] - positions[atom2_idx, 0] 
        dy2 = positions[atom3_idx, 1] - positions[atom2_idx, 1]
        dz2 = positions[atom3_idx, 2] - positions[atom2_idx, 2]
        
        # Magnitudes
        r1 = np.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
        r2 = np.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
        
        if r1 > 1e-10 and r2 > 1e-10:
            # Dot product
            dot_prod = dx1*dx2 + dy1*dy2 + dz1*dz2
            cos_theta = dot_prod / (r1 * r2)
            cos_theta = max(-1.0, min(1.0, cos_theta))  # Clamp
            
            theta = np.arccos(cos_theta)
            sin_theta = np.sin(theta)
            
            if abs(sin_theta) > 1e-10:
                dtheta = theta - theta0
                force_const = -K * dtheta / sin_theta
                
                # Force calculation (simplified)
                coeff1 = force_const / (r1 * r1)
                coeff2 = force_const / (r2 * r2)
                
                f1x = coeff1 * (dx2/r2 - cos_theta * dx1/r1)
                f1y = coeff1 * (dy2/r2 - cos_theta * dy1/r1) 
                f1z = coeff1 * (dz2/r2 - cos_theta * dz1/r1)
                
                f3x = coeff2 * (dx1/r1 - cos_theta * dx2/r2)
                f3y = coeff2 * (dy1/r1 - cos_theta * dy2/r2)
                f3z = coeff2 * (dz1/r1 - cos_theta * dz2/r2)
                
                # Apply forces
                forces[atom1_idx, 0] += f1x
                forces[atom1_idx, 1] += f1y
                forces[atom1_idx, 2] += f1z
                
                forces[atom2_idx, 0] -= (f1x + f3x)
                forces[atom2_idx, 1] -= (f1y + f3y)
                forces[atom2_idx, 2] -= (f1z + f3z)
                
                forces[atom3_idx, 0] += f3x
                forces[atom3_idx, 1] += f3y
                forces[atom3_idx, 2] += f3z

@njit
def numba_nonbonded_forces(positions, forces, neighbor_list, charges, sigma, epsilon):
    """Pure NumPy non-bonded force calculation with Numba"""
    n_pairs = neighbor_list.shape[0]
    
    for i in range(n_pairs):
        atom1_idx = neighbor_list[i, 0]
        atom2_idx = neighbor_list[i, 1]
        
        if atom1_idx != atom2_idx:
            # Distance vector
            dx = positions[atom2_idx, 0] - positions[atom1_idx, 0]
            dy = positions[atom2_idx, 1] - positions[atom1_idx, 1]
            dz = positions[atom2_idx, 2] - positions[atom1_idx, 2]
            
            r2 = dx*dx + dy*dy + dz*dz
            r = np.sqrt(r2)
            
            if r > 0.1:  # Avoid singularity
                # Lennard-Jones
                sig_r = sigma / r
                sig_r6 = sig_r**6
                sig_r12 = sig_r6 * sig_r6
                
                lj_force = 24.0 * epsilon * (2.0 * sig_r12 - sig_r6) / r
                
                # Coulomb
                q1q2 = charges[atom1_idx] * charges[atom2_idx]
                coulomb_force = 332.0637 * q1q2 / r2  # 332.0637 converts to kcal/mol
                
                total_force = (lj_force + coulomb_force) / r
                
                # Force components
                fx = total_force * dx
                fy = total_force * dy
                fz = total_force * dz
                
                # Apply forces
                forces[atom1_idx, 0] += fx
                forces[atom1_idx, 1] += fy
                forces[atom1_idx, 2] += fz
                
                forces[atom2_idx, 0] -= fx
                forces[atom2_idx, 1] -= fy
                forces[atom2_idx, 2] -= fz

@njit(parallel=True)
def numba_integrate(positions, velocities, forces, masses, dt):
    """Pure NumPy integration with Numba"""
    n_atoms = positions.shape[0]
    
    for i in prange(n_atoms):
        mass = masses[i]
        inv_mass = 1.0 / mass if mass > 0 else 0.0
        
        # Velocity Verlet integration
        for j in range(3):
            acc = forces[i, j] * inv_mass
            velocities[i, j] += 0.5 * acc * dt
            positions[i, j] += velocities[i, j] * dt

def extract_data_from_system(sysobj):
    """Convert System object to pure NumPy arrays for Numba"""
    # Count total atoms
    n_atoms = sum(len(mol.atoms) for mol in sysobj.molecules)
    n_molecules = len(sysobj.molecules)
    
    # Extract positions, velocities, forces, masses, charges
    positions = np.zeros((n_atoms, 3))
    velocities = np.zeros((n_atoms, 3)) 
    forces = np.zeros((n_atoms, 3))
    masses = np.zeros(n_atoms)
    charges = np.zeros(n_atoms)
    
    atom_idx = 0
    for mol in sysobj.molecules:
        for atom in mol.atoms:
            positions[atom_idx] = atom.p
            velocities[atom_idx] = atom.v
            forces[atom_idx] = atom.f
            masses[atom_idx] = atom.mass
            charges[atom_idx] = atom.charge
            atom_idx += 1
    
    # Extract bond information
    bonds = []
    K_bond = []
    L0_bond = []
    
    atom_offset = 0
    for mol in sysobj.molecules:
        for bond in mol.bonds:
            # Convert local atom indices to global indices
            bonds.append([atom_offset + bond.a1, atom_offset + bond.a2])
            K_bond.append(bond.K)
            L0_bond.append(bond.L0)
        atom_offset += len(mol.atoms)
    
    bonds = np.array(bonds, dtype=np.int32) if bonds else np.zeros((0, 2), dtype=np.int32)
    K_bond = np.array(K_bond) if K_bond else np.zeros(0)
    L0_bond = np.array(L0_bond) if L0_bond else np.zeros(0)
    
    # Extract angle information
    angles = []
    K_angle = []
    theta0_angle = []
    
    atom_offset = 0
    for mol in sysobj.molecules:
        for angle in mol.angles:
            # Convert local atom indices to global indices
            angles.append([atom_offset + angle.a1, atom_offset + angle.a2, atom_offset + angle.a3])
            K_angle.append(angle.K)
            theta0_angle.append(angle.Phi0)  # Fixed: use Phi0 not theta0
        atom_offset += len(mol.atoms)
    
    angles = np.array(angles, dtype=np.int32) if angles else np.zeros((0, 3), dtype=np.int32)
    K_angle = np.array(K_angle) if K_angle else np.zeros(0)
    theta0_angle = np.array(theta0_angle) if theta0_angle else np.zeros(0)
    
    # Extract neighbor list - use molecule-level neighbors
    neighbor_pairs = []
    
    for i, mol_i in enumerate(sysobj.molecules):
        for j in mol_i.neighbours:
            if i < j:  # Avoid double counting
                mol_j = sysobj.molecules[j]
                # Add all atom pairs between neighboring molecules
                atom_offset_i = sum(len(sysobj.molecules[k].atoms) for k in range(i))
                atom_offset_j = sum(len(sysobj.molecules[k].atoms) for k in range(j))
                
                for ai, atom_i in enumerate(mol_i.atoms):
                    for aj, atom_j in enumerate(mol_j.atoms):
                        neighbor_pairs.append([atom_offset_i + ai, atom_offset_j + aj])
    
    neighbor_list = np.array(neighbor_pairs, dtype=np.int32) if neighbor_pairs else np.zeros((0, 2), dtype=np.int32)
    
    # LJ parameters (simplified - using water values)
    sigma = 3.15  # Angstrom
    epsilon = 0.1521  # kcal/mol
    
    return {
        'positions': positions,
        'velocities': velocities, 
        'forces': forces,
        'masses': masses,
        'charges': charges,
        'bonds': bonds,
        'K_bond': K_bond,
        'L0_bond': L0_bond,
        'angles': angles,
        'K_angle': K_angle,
        'theta0_angle': theta0_angle,
        'neighbor_list': neighbor_list,
        'sigma': sigma,
        'epsilon': epsilon
    }

def update_system_from_data(sysobj, data):
    """Update System object from NumPy arrays"""
    atom_idx = 0
    for mol in sysobj.molecules:
        for atom in mol.atoms:
            atom.p = data['positions'][atom_idx].copy()
            atom.v = data['velocities'][atom_idx].copy()
            atom.f = data['forces'][atom_idx].copy()
            atom_idx += 1

# Test function
def run_pure_numba_simulation(N_mol, steps, dt):
    """Run simulation using pure Numba functions"""
    # Import here to avoid circular imports
    from Water_sequential import MakeWater, Sim_Configuration, BuildNeighborList
    
    # Create system
    original_argv = sys.argv.copy()
    sys.argv = ['test', str(N_mol), str(steps), str(dt), '/tmp/dummy.txt']
    sc = Sim_Configuration(sys.argv)
    sysobj = MakeWater(sc.no_mol)
    sys.argv = original_argv
    
    start_time = time.perf_counter()
    
    for step in range(steps):
        # Build neighbor list every 100 steps
        if step % 100 == 0:
            BuildNeighborList(sysobj)
        
        # Extract data for Numba
        data = extract_data_from_system(sysobj)
        
        # Clear forces
        data['forces'].fill(0.0)
        
        # Run Numba-compiled force calculations
        if len(data['bonds']) > 0:
            numba_bond_forces(data['positions'], data['forces'], data['bonds'], 
                            data['K_bond'], data['L0_bond'])
        
        if len(data['angles']) > 0:
            numba_angle_forces(data['positions'], data['forces'], data['angles'],
                             data['K_angle'], data['theta0_angle'])
        
        if len(data['neighbor_list']) > 0:
            numba_nonbonded_forces(data['positions'], data['forces'], data['neighbor_list'],
                                 data['charges'], data['sigma'], data['epsilon'])
        
        # Integration
        numba_integrate(data['positions'], data['velocities'], data['forces'], 
                       data['masses'], dt)
        
        # Update system object
        update_system_from_data(sysobj, data)
    
    end_time = time.perf_counter()
    return end_time - start_time

def warmup_pure_numba():
    """Warmup pure Numba functions"""
    print("Warming up pure Numba functions...")
    
    # Create small test data
    pos = np.random.random((10, 3))
    forces = np.zeros((10, 3))
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    K_bond = np.array([450.0, 450.0])
    L0_bond = np.array([1.0, 1.0])
    
    # Compile functions
    numba_bond_forces(pos, forces, bonds, K_bond, L0_bond)
    print("Pure Numba compilation complete!")

if __name__ == "__main__":
    # Test the pure Numba implementation
    warmup_pure_numba()
    
    print("\nTesting pure Numba implementation:")
    for size in [100, 500, 1000]:
        time_taken = run_pure_numba_simulation(size, 100, 0.001)
        print(f"Size {size}: {time_taken:.3f} seconds")