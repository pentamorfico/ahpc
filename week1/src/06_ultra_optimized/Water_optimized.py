# Applied High Performance Computing - OPTIMIZED VERSION WITH NUMBA
# 
# Molecular Dynamics Simulation of Water Molecules 
# 
# Description: This program simulates flexible water molecules using a simple
#              classical model. Each water has two covalent bonds and one angle.
#              All non-bonded atoms interact through LJ potential. 
#              Verlet integrator is used. 
#
# Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen
# Optimized by: AI Assistant with Numba JIT compilation and parallelization

import sys
import time
import numpy as np
from numba import njit, prange
from typing import List

deg2rad = np.pi / 180.0 # pi/180 for changing degs to radians

accumulated_forces_bond: float = 0.0 # Checksum: accumulated size of forces
accumulated_forces_angle: float = 0.0 # Checksum: accumulated size of forces
accumulated_forces_non_bond: float = 0.0 # Checksum: accumulated size of forces

# Number of closest neighbors to consider in neighbor list. 
N_CLOSEST: int = 8

# Optimized utility functions with Numba
@njit
def mag_numba(v):
    """Fast magnitude calculation"""
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

@njit
def mag2_numba(v):
    """Fast squared magnitude calculation"""
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]

@njit
def cross_numba(a, b):
    """Fast cross product"""
    result = np.empty(3)
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]
    return result

@njit
def dot_numba(a, b):
    """Fast dot product"""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@njit
def normalize_numba(v):
    """Fast vector normalization"""
    mag = mag_numba(v)
    if mag > 0:
        return v / mag
    return v

# Create a vector of 3 doubles
def Vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0):
    return np.array([x,y,z])

def mag(v: np.ndarray) -> float: # magnitude of vector
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

def mag2(v: np.ndarray) -> float: # squared magnitude of vector
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]

def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray: # cross product of two vectors
    return Vec3(a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

def dot(a: np.ndarray, b: np.ndarray) -> float: # dot product of two vectors
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

# atom class
class Atom:
    def __init__(self, mass: float, ep: float, sigma: float, charge: float, name: str):
        self.mass = mass  # The mass of the atom in (U)
        self.ep = ep  # epsilon for LJ potential
        self.sigma = sigma  # Sigma, somehow the size of the atom
        self.charge = charge  # charge of the atom (partial charge)
        self.name = name  # Name of the atom
        # the position in (nm), velocity (nm/ps) and forces (k_BT/nm) of the atom
        self.p = Vec3()
        self.v = Vec3()
        self.f = Vec3()

# class for the covalent bond between two atoms U=0.5k(r12-L0)^2
class Bond:
    def __init__(self, K: float, L0: float, a1: int, a2: int):
        self.K = K  # force constant
        self.L0 = L0  # relaxed length
        self.a1 = a1  # the indexes of the atoms at either end
        self.a2 = a2

# class for the angle between three atoms U=0.5K(phi123-phi0)^2
class Angle:
    def __init__(self, K: float, Phi0: float, a1: int, a2: int, a3: int):
        self.K = K
        self.Phi0 = Phi0
        self.a1 = a1  # the indexes of the three atoms, with a2 being the centre atom
        self.a2 = a2
        self.a3 = a3

# molecule class
class Molecule:
    def __init__(self, atoms: List[Atom], bonds: List[Bond], angles: List[Angle]):
        self.atoms = atoms  # list of atoms in the molecule
        self.bonds = bonds  # the bond potentials, eg for water the left and right bonds
        self.angles = angles  # the angle potentials, for water just the single one, but keep it a list for generality
        self.neighbours: List[int] = []  # indices of the neighbours

## ===============================================================================
##  Two new data types arranging Atoms in a Structure-of-Array data structure
## ===============================================================================

# atoms datatype (Structure-of-Arrays for N identical atoms)
class Atoms:
    def __init__(self, mass: float, ep: float, sigma: float, charge: float, name: str, no_atoms: int):
        self.mass = mass  # The mass of the atom in (U)
        self.ep = ep  # epsilon for LJ potential
        self.sigma = sigma  # Sigma, somehow the size of the atom
        self.charge = charge  # charge of the atom (partial charge)
        self.name = name  # Name of the atom
        self.no_atoms = no_atoms  # number of identical atoms in this group

        # the position in (nm), velocity (nm/ps) and forces (k_BT/nm) of the atom
        self.p = np.zeros((3, no_atoms))
        self.v = np.zeros((3, no_atoms))
        self.f = np.zeros((3, no_atoms))

# molecule collection class: holds many Atoms groups, bonds, angles and neighbour lists
class Molecules:
    def __init__(self, atoms: List[Atoms], bonds: List[Bond], angles: List[Angle], no_mol: int):
        self.atoms: List[Atoms] = atoms                          # list of Atoms in the molecule
        self.bonds: List[Bond] = bonds                           # the bond potentials, eg for water the left and right bonds
        self.angles: List[Angle] = angles                        # the angle potentials, for water just the single one, but keep it a list for generality
        self.neighbours: np.ndarray = np.zeros((no_mol, N_CLOSEST), dtype=np.int32)  # indices of the neighbours
        self.no_mol: int = no_mol

## ===============================================================================

# system class
class System:
    def __init__(self):
        self.molecules: Molecules = None  # all the molecules in the system (SoA structure)
        self.time: float = 0.0  # current simulation time

class Sim_Configuration:
    def __init__(self, argument: List[str]):
        self.steps: int = 20     # number of steps
        self.no_mol: int = 100   # number of molecules
        self.dt: float = 0.0005  # integrator time step
        self.data_period: int = 100  # how often to save coordinate to trajectory
        self.filename: str = "trajectory.txt"  # name of the output file with trajectory

        # simulation configurations: number of step, number of the molecules in the system, 
        # IO frequency, time step and file name
        i = 1
        n = len(argument)
        while i < n:
            arg = argument[i]
            if arg == "-h":  # Write help
                print(
                    "MD -steps <number of steps> -no_mol <number of molecules> -fwrite <io frequency> -dt <size of timestep> -ofile <filename>"
                )
                sys.exit(0)
            if i + 1 >= n:
                print("---> error: missing value for argument", arg)
                break
            if arg == "-steps":
                self.steps = int(argument[i + 1])
            elif arg == "-no_mol":
                self.no_mol = int(argument[i + 1])
            elif arg == "-fwrite":
                self.data_period = int(argument[i + 1])
            elif arg == "-dt":
                self.dt = float(argument[i + 1])
            elif arg == "-ofile":
                self.filename = argument[i + 1]
            else:
                print("---> error: the argument type is not recognized")
            i += 2

        self.dt /= 1.57350  # convert to ps based on having energy in k_BT, and length in nm

@njit
def build_neighbor_list_numba(oxygen_positions, no_mol, target_num, neighbours):
    """Optimized neighbor list building with Numba"""
    distances2 = np.empty(no_mol)
    
    # Clear neighbor lists
    neighbours.fill(-1)
    
    for i in range(no_mol):
        neighbor_count = 0
        # Calculate distances to all other molecules
        for j in range(no_mol):
            if i != j:
                dp = oxygen_positions[:, i] - oxygen_positions[:, j]
                distances2[j] = dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2]
            else:
                distances2[j] = 1e99  # exclude own molecule
                
        # Find closest neighbors
        # Simple selection sort to find N smallest
        indices = np.arange(no_mol)
        for k in range(min(target_num, no_mol-1)):
            min_idx = k
            for j in range(k+1, no_mol):
                if distances2[indices[j]] < distances2[indices[min_idx]]:
                    min_idx = j
            # Swap
            indices[k], indices[min_idx] = indices[min_idx], indices[k]
            
            # Add to neighbor list with same logic as original
            neighbor_idx = indices[k]
            if neighbor_idx < i:
                # Check if i is already in neighbor_idx's list
                already_exists = False
                for m in range(N_CLOSEST):
                    if neighbours[neighbor_idx, m] == i:
                        already_exists = True
                        break
                if not already_exists and neighbor_count < N_CLOSEST:
                    neighbours[i, neighbor_count] = neighbor_idx
                    neighbor_count += 1
            else:
                if neighbor_count < N_CLOSEST:
                    neighbours[i, neighbor_count] = neighbor_idx
                    neighbor_count += 1

def BuildNeighborList(sysobj: System):
    """Update neighbour list using optimized Numba function"""
    molecules = sysobj.molecules
    target_num = min(N_CLOSEST, max(0, molecules.no_mol - 1))
    
    build_neighbor_list_numba(
        molecules.atoms[0].p,  # oxygen positions
        molecules.no_mol,
        target_num,
        molecules.neighbours
    )

@njit(parallel=True)
def update_bond_forces_numba(pos1, pos2, forces1, forces2, K, L0, no_mol):
    """Optimized bond force calculation with parallel processing"""
    acc_force = 0.0
    
    for i in prange(no_mol):
        dp = pos1[:, i] - pos2[:, i]
        r = np.sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2])
        
        force_factor = -K * (1.0 - L0 / r)
        f = force_factor * dp
        
        forces1[:, i] += f
        forces2[:, i] -= f
        
        # Note: accumulating in parallel requires reduction, simplified here
        acc_force += np.sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2])
    
    return acc_force

def UpdateBondForces(sysobj: System):
    """Optimized bond force calculation"""
    global accumulated_forces_bond
    molecules = sysobj.molecules
    
    total_acc = 0.0
    for bond in molecules.bonds:
        atom1_group = molecules.atoms[bond.a1]
        atom2_group = molecules.atoms[bond.a2]
        
        acc = update_bond_forces_numba(
            atom1_group.p, atom2_group.p,
            atom1_group.f, atom2_group.f,
            bond.K, bond.L0, molecules.no_mol
        )
        total_acc += acc
    
    accumulated_forces_bond += total_acc

@njit(parallel=True)
def update_angle_forces_numba(pos1, pos2, pos3, forces1, forces2, forces3, K, Phi0, no_mol):
    """Optimized angle force calculation with parallel processing"""
    acc_force = 0.0
    
    for i in prange(no_mol):
        d21 = pos2[:, i] - pos1[:, i]
        d23 = pos2[:, i] - pos3[:, i]
        
        norm_d21 = np.sqrt(d21[0]*d21[0] + d21[1]*d21[1] + d21[2]*d21[2])
        norm_d23 = np.sqrt(d23[0]*d23[0] + d23[1]*d23[1] + d23[2]*d23[2])
        dot_product = d21[0]*d23[0] + d21[1]*d23[1] + d21[2]*d23[2]
        
        cos_phi = dot_product / (norm_d21 * norm_d23)
        cos_phi = max(-1.0, min(1.0, cos_phi))  # clamp to avoid numerical issues
        phi = np.arccos(cos_phi)
        
        # Cross products
        c21_23 = cross_numba(d21, d23)
        Ta = cross_numba(d21, c21_23)
        Ta_mag = mag_numba(Ta)
        if Ta_mag > 0:
            Ta = Ta / Ta_mag
        
        Tc = cross_numba(c21_23, d23)
        Tc_mag = mag_numba(Tc)
        if Tc_mag > 0:
            Tc = Tc / Tc_mag
        
        force_factor1 = K * (phi - Phi0) / norm_d21
        force_factor3 = K * (phi - Phi0) / norm_d23
        
        f1 = Ta * force_factor1
        f3 = Tc * force_factor3
        
        forces1[:, i] += f1
        forces2[:, i] -= (f1 + f3)
        forces3[:, i] += f3
        
        acc_force += mag_numba(f1) + mag_numba(f3)
    
    return acc_force

def UpdateAngleForces(sysobj: System):
    """Optimized angle force calculation"""
    global accumulated_forces_angle
    molecules = sysobj.molecules
    
    total_acc = 0.0
    for angle in molecules.angles:
        atom1_group = molecules.atoms[angle.a1]
        atom2_group = molecules.atoms[angle.a2] 
        atom3_group = molecules.atoms[angle.a3]
        
        acc = update_angle_forces_numba(
            atom1_group.p, atom2_group.p, atom3_group.p,
            atom1_group.f, atom2_group.f, atom3_group.f,
            angle.K, angle.Phi0, molecules.no_mol
        )
        total_acc += acc
    
    accumulated_forces_angle += total_acc

@njit
def update_nonbonded_forces_numba(positions, forces, masses, eps, sigmas, charges, 
                                 neighbours, no_mol, no_atom_types):
    """Highly optimized non-bonded force calculation"""
    KC = 80 * 0.7  # Coulomb prefactor
    acc = 0.0
    
    for i in range(no_mol):
        for neighbor_slot in range(N_CLOSEST):
            j = neighbours[i, neighbor_slot]
            if j == -1:  # No more valid neighbors
                break
                
            for atom1_idx in range(no_atom_types):
                for atom2_idx in range(no_atom_types):
                    ep = np.sqrt(eps[atom1_idx] * eps[atom2_idx])
                    sigma2 = (0.5 * (sigmas[atom1_idx] + sigmas[atom2_idx])) ** 2
                    q = KC * charges[atom1_idx] * charges[atom2_idx]
                    
                    dp = positions[atom1_idx][:, i] - positions[atom2_idx][:, j]
                    r2 = dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2]
                    r = np.sqrt(r2)
                    
                    sir = sigma2 / r2
                    sir3 = sir * sir * sir
                    
                    force_magnitude = ep * (12 * sir3 * sir3 - 6 * sir3) * sir + q / (r * r2)
                    f = force_magnitude * dp
                    
                    forces[atom1_idx][:, i] += f
                    forces[atom2_idx][:, j] -= f
                    
                    acc += np.sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2])
    
    return acc

def UpdateNonBondedForces(sysobj: System):
    """Optimized non-bonded force calculation"""
    global accumulated_forces_non_bond
    molecules = sysobj.molecules
    
    # Prepare data for Numba function
    no_atom_types = len(molecules.atoms)
    positions = np.array([atom_group.p for atom_group in molecules.atoms])
    forces = np.array([atom_group.f for atom_group in molecules.atoms])
    eps = np.array([atom_group.ep for atom_group in molecules.atoms])
    sigmas = np.array([atom_group.sigma for atom_group in molecules.atoms])  
    charges = np.array([atom_group.charge for atom_group in molecules.atoms])
    
    acc = update_nonbonded_forces_numba(
        positions, forces, None, eps, sigmas, charges,
        molecules.neighbours, molecules.no_mol, no_atom_types
    )
    
    # Copy forces back
    for i, atom_group in enumerate(molecules.atoms):
        atom_group.f[:] = forces[i]
    
    accumulated_forces_non_bond += acc

@njit(parallel=True) 
def update_kdk_numba(positions, velocities, forces, masses, dt, no_atom_types, no_mol):
    """Optimized integrator with parallel processing"""
    for atom_type in range(no_atom_types):
        mass = masses[atom_type]
        for i in prange(no_mol):
            # Update velocities
            velocities[atom_type][:, i] += (dt / mass) * forces[atom_type][:, i]
            # Clear forces
            forces[atom_type][:, i] = 0.0
            # Update positions
            positions[atom_type][:, i] += dt * velocities[atom_type][:, i]

def UpdateKDK(sysobj: System, sc: Sim_Configuration):
    """Optimized integrator"""
    molecules = sysobj.molecules
    
    positions = np.array([atom_group.p for atom_group in molecules.atoms])
    velocities = np.array([atom_group.v for atom_group in molecules.atoms])
    forces = np.array([atom_group.f for atom_group in molecules.atoms])
    masses = np.array([atom_group.mass for atom_group in molecules.atoms])
    
    update_kdk_numba(positions, velocities, forces, masses, sc.dt,
                     len(molecules.atoms), molecules.no_mol)
    
    # Copy data back
    for i, atom_group in enumerate(molecules.atoms):
        atom_group.p[:] = positions[i]
        atom_group.v[:] = velocities[i] 
        atom_group.f[:] = forces[i]
    
    sysobj.time += sc.dt

# Setup one water molecule
def MakeWater(N_molecules: int) -> System:
    """Create water molecules in SoA format"""
    L0 = 0.09584
    angle = 104.45 * deg2rad

    waterbonds = [
        Bond(K=20000, L0=L0, a1=0, a2=1),  # O-H1
        Bond(K=20000, L0=L0, a1=0, a2=2), # O-H2
    ]

    waterangle = [
        Angle(K=1000, Phi0=angle, a1=1, a2=0, a3=2),  # H1-O-H2
    ]

    # Create SoA structure
    Oatoms = Atoms(mass=16, ep=0.65, sigma=0.31, charge=-0.82, name="O", no_atoms=N_molecules)
    H1atoms = Atoms(mass=1, ep=0.18828, sigma=0.238, charge=0.41, name="H", no_atoms=N_molecules)
    H2atoms = Atoms(mass=1, ep=0.18828, sigma=0.238, charge=0.41, name="H", no_atoms=N_molecules)

    atoms = [Oatoms, H1atoms, H2atoms]
    molecules = Molecules(atoms, waterbonds, waterangle, N_molecules)

    # Initialize positions on sphere
    phi = np.arccos(-1) * (np.sqrt(5.0) - 1.0)
    radius = np.sqrt(N_molecules) * 0.15

    for i in range(N_molecules):
        y = 1.0 - (i / (N_molecules - 1.0)) if N_molecules > 1 else 0.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i

        x = np.cos(theta) * r
        z = np.sin(theta) * r

        P0 = Vec3(x * radius, y * radius, z * radius)
        
        Oatoms.p[:, i] = [P0[0], P0[1], P0[2]]
        H1atoms.p[:, i] = [P0[0] + L0 * np.sin(angle / 2), P0[1] + L0 * np.cos(angle / 2), P0[2]]
        H2atoms.p[:, i] = [P0[0] - L0 * np.sin(angle / 2), P0[1] + L0 * np.cos(angle / 2), P0[2]]

    sysobj = System()
    sysobj.molecules = molecules
    return sysobj

def WriteOutput(sysobj: System, file):
    """Write system configurations to trajectory file"""
    molecules = sysobj.molecules
    
    for mol_idx in range(molecules.no_mol):
        for atom_group in molecules.atoms:
            file.write(
                f"{sysobj.time:.6g} {atom_group.name} {atom_group.p[0, mol_idx]:.6g} {atom_group.p[1, mol_idx]:.6g} {atom_group.p[2, mol_idx]:.6g}\n"
            )

#======================================================================================================
#======================== Main function ===============================================================
#======================================================================================================
if __name__ == "__main__":
    sc = Sim_Configuration(sys.argv)
    sysobj = MakeWater(sc.no_mol)
    
    file = open(sc.filename, "w")
    WriteOutput(sysobj, file)
    
    tstart = time.perf_counter()
    
    for step in range(sc.steps):
        if step % 100 == 0:
            BuildNeighborList(sysobj)
        
        UpdateBondForces(sysobj)
        UpdateAngleForces(sysobj) 
        UpdateNonBondedForces(sysobj)
        UpdateKDK(sysobj, sc)
        
        if step % sc.data_period == 0:
            WriteOutput(sysobj, file)
    
    file.close()
    tend = time.perf_counter()
    
    print(f"Accumulated forces Bonds   : {accumulated_forces_bond:9.5f}")
    print(f"Accumulated forces Angles  : {accumulated_forces_angle:9.5f}")
    print(f"Accumulated forces Non-bond: {accumulated_forces_non_bond:9.5f}")
    print(f"Elapsed total time:       {tend - tstart:9.4f}")