#!/usr/bin/env python3
import numpy as np
import time
import sys
from numba import njit, prange, types
from numba.typed import List

# Add explicit type signatures to eliminate "argument type not recognized" errors
@njit('void(float64[:,:], float64[:,:], int32[:,:], float64[:], float64[:])', parallel=True)
def numba_bond_forces_typed(positions, forces, bonds, K_bond, L0_bond):
    """Explicitly typed bond force calculation with Numba"""
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
        
        if r > 1e-10:
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

@njit('void(float64[:,:], float64[:,:], int32[:,:], float64[:], float64[:])', parallel=True)
def numba_angle_forces_typed(positions, forces, angles, K_angle, theta0_angle):
    """Explicitly typed angle force calculation with Numba"""
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

@njit('void(float64[:,:], float64[:,:], int32[:,:], float64[:], float64, float64)')
def numba_nonbonded_forces_typed(positions, forces, neighbor_list, charges, sigma, epsilon):
    """Explicitly typed non-bonded force calculation with Numba"""
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

@njit('void(float64[:,:], float64[:,:], float64[:,:], float64[:], float64)', parallel=True)
def numba_integrate_typed(positions, velocities, forces, masses, dt):
    """Explicitly typed integration with Numba"""
    n_atoms = positions.shape[0]
    
    for i in prange(n_atoms):
        mass = masses[i]
        inv_mass = 1.0 / mass if mass > 0 else 0.0
        
        # Velocity Verlet integration
        for j in range(3):
            acc = forces[i, j] * inv_mass
            velocities[i, j] += 0.5 * acc * dt
            positions[i, j] += velocities[i, j] * dt

def extract_typed_data_from_system(sysobj):
    """Convert System object to properly typed NumPy arrays"""
    # Count total atoms
    n_atoms = sum(len(mol.atoms) for mol in sysobj.molecules)
    
    # Extract with explicit types
    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    velocities = np.zeros((n_atoms, 3), dtype=np.float64) 
    forces = np.zeros((n_atoms, 3), dtype=np.float64)
    masses = np.zeros(n_atoms, dtype=np.float64)
    charges = np.zeros(n_atoms, dtype=np.float64)
    
    atom_idx = 0
    for mol in sysobj.molecules:
        for atom in mol.atoms:
            positions[atom_idx] = atom.p
            velocities[atom_idx] = atom.v
            forces[atom_idx] = atom.f
            masses[atom_idx] = atom.mass
            charges[atom_idx] = atom.charge
            atom_idx += 1
    
    # Extract bond information with explicit types
    bonds_list = []
    K_bond_list = []
    L0_bond_list = []
    
    atom_offset = 0
    for mol in sysobj.molecules:
        for bond in mol.bonds:
            bonds_list.append([atom_offset + bond.a1, atom_offset + bond.a2])
            K_bond_list.append(bond.K)
            L0_bond_list.append(bond.L0)
        atom_offset += len(mol.atoms)
    
    bonds = np.array(bonds_list, dtype=np.int32) if bonds_list else np.zeros((0, 2), dtype=np.int32)
    K_bond = np.array(K_bond_list, dtype=np.float64) if K_bond_list else np.zeros(0, dtype=np.float64)
    L0_bond = np.array(L0_bond_list, dtype=np.float64) if L0_bond_list else np.zeros(0, dtype=np.float64)
    
    # Extract angle information with explicit types
    angles_list = []
    K_angle_list = []
    theta0_angle_list = []
    
    atom_offset = 0
    for mol in sysobj.molecules:
        for angle in mol.angles:
            angles_list.append([atom_offset + angle.a1, atom_offset + angle.a2, atom_offset + angle.a3])
            K_angle_list.append(angle.K)
            theta0_angle_list.append(angle.Phi0)
        atom_offset += len(mol.atoms)
    
    angles = np.array(angles_list, dtype=np.int32) if angles_list else np.zeros((0, 3), dtype=np.int32)
    K_angle = np.array(K_angle_list, dtype=np.float64) if K_angle_list else np.zeros(0, dtype=np.float64)
    theta0_angle = np.array(theta0_angle_list, dtype=np.float64) if theta0_angle_list else np.zeros(0, dtype=np.float64)
    
    # Extract neighbor list with explicit types
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
    
    # LJ parameters with explicit types
    sigma = np.float64(3.15)  # Angstrom
    epsilon = np.float64(0.1521)  # kcal/mol
    
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

def update_system_from_typed_data(sysobj, data):
    """Update System object from typed NumPy arrays"""
    atom_idx = 0
    for mol in sysobj.molecules:
        for atom in mol.atoms:
            atom.p = data['positions'][atom_idx].copy()
            atom.v = data['velocities'][atom_idx].copy()
            atom.f = data['forces'][atom_idx].copy()
            atom_idx += 1

def run_typed_numba_simulation(N_mol, steps, dt):
    """Run simulation using explicitly typed Numba functions"""
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
        
        # Extract typed data for Numba
        data = extract_typed_data_from_system(sysobj)
        
        # Clear forces
        data['forces'].fill(0.0)
        
        # Run Numba-compiled force calculations with explicit types
        if len(data['bonds']) > 0:
            numba_bond_forces_typed(data['positions'], data['forces'], data['bonds'], 
                                   data['K_bond'], data['L0_bond'])
        
        if len(data['angles']) > 0:
            numba_angle_forces_typed(data['positions'], data['forces'], data['angles'],
                                    data['K_angle'], data['theta0_angle'])
        
        if len(data['neighbor_list']) > 0:
            numba_nonbonded_forces_typed(data['positions'], data['forces'], data['neighbor_list'],
                                        data['charges'], data['sigma'], data['epsilon'])
        
        # Integration
        numba_integrate_typed(data['positions'], data['velocities'], data['forces'], 
                             data['masses'], np.float64(dt))
        
        # Update system object
        update_system_from_typed_data(sysobj, data)
    
    end_time = time.perf_counter()
    return end_time - start_time

def warmup_typed_numba():
    """Warmup explicitly typed Numba functions"""
    print("Warming up typed Numba functions...")
    
    # Create small test data with explicit types
    pos = np.random.random((10, 3)).astype(np.float64)
    forces = np.zeros((10, 3), dtype=np.float64)
    vel = np.zeros((10, 3), dtype=np.float64)
    masses = np.ones(10, dtype=np.float64)
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    K_bond = np.array([450.0, 450.0], dtype=np.float64)
    L0_bond = np.array([1.0, 1.0], dtype=np.float64)
    charges = np.zeros(10, dtype=np.float64)
    neighbor_list = np.array([[0, 1], [1, 2]], dtype=np.int32)
    
    # Compile functions
    numba_bond_forces_typed(pos, forces, bonds, K_bond, L0_bond)
    numba_integrate_typed(pos, vel, forces, masses, np.float64(0.001))
    numba_nonbonded_forces_typed(pos, forces, neighbor_list, charges, np.float64(3.15), np.float64(0.1521))
    
    print("Typed Numba compilation complete!")

if __name__ == "__main__":
    # Test the typed Numba implementation
    warmup_typed_numba()
    
    print("\nTesting typed Numba implementation:")
    for size in [100, 500, 1000]:
        time_taken = run_typed_numba_simulation(size, 100, 0.001)
        print(f"Size {size}: {time_taken:.3f} seconds")