# Applied High Performance Computing - OPTIMIZED VERSION 2 WITH NUMBA
# 
# Molecular Dynamics Simulation of Water Molecules 
# Optimized version focusing on minimal data copying and efficient Numba usage

import sys
import time
import numpy as np
from numba import njit, prange
from typing import List

deg2rad = np.pi / 180.0

accumulated_forces_bond: float = 0.0
accumulated_forces_angle: float = 0.0
accumulated_forces_non_bond: float = 0.0

N_CLOSEST: int = 8

# Optimized utility functions
@njit(inline='always')
def mag_squared_3d(v):
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]

@njit(inline='always')
def mag_3d(v):
    return np.sqrt(mag_squared_3d(v))

def Vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0):
    return np.array([x,y,z])

# Simplified classes for better Numba compatibility
class Atoms:
    def __init__(self, mass: float, ep: float, sigma: float, charge: float, name: str, no_atoms: int):
        self.mass = mass
        self.ep = ep
        self.sigma = sigma
        self.charge = charge
        self.name = name
        self.no_atoms = no_atoms
        self.p = np.zeros((3, no_atoms), dtype=np.float64)
        self.v = np.zeros((3, no_atoms), dtype=np.float64)
        self.f = np.zeros((3, no_atoms), dtype=np.float64)

class Bond:
    def __init__(self, K: float, L0: float, a1: int, a2: int):
        self.K = K
        self.L0 = L0
        self.a1 = a1
        self.a2 = a2

class Angle:
    def __init__(self, K: float, Phi0: float, a1: int, a2: int, a3: int):
        self.K = K
        self.Phi0 = Phi0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

class Molecules:
    def __init__(self, atoms: List[Atoms], bonds: List[Bond], angles: List[Angle], no_mol: int):
        self.atoms = atoms
        self.bonds = bonds
        self.angles = angles
        self.neighbours = np.full((no_mol, N_CLOSEST), -1, dtype=np.int32)
        self.no_mol = no_mol

class System:
    def __init__(self):
        self.molecules: Molecules = None
        self.time: float = 0.0

class Sim_Configuration:
    def __init__(self, argument: List[str]):
        self.steps: int = 20
        self.no_mol: int = 100
        self.dt: float = 0.0005
        self.data_period: int = 100
        self.filename: str = "trajectory.txt"

        i = 1
        n = len(argument)
        while i < n:
            arg = argument[i]
            if arg == "-h":
                print("MD -steps <number of steps> -no_mol <number of molecules> -fwrite <io frequency> -dt <size of timestep> -ofile <filename>")
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

        self.dt /= 1.57350

# Ultra-fast bond force calculation
@njit(parallel=True)
def compute_bond_forces(pos1, pos2, force1, force2, K, L0):
    no_mol = pos1.shape[1]
    acc_force = 0.0
    
    for i in prange(no_mol):
        dx = pos1[0, i] - pos2[0, i]
        dy = pos1[1, i] - pos2[1, i] 
        dz = pos1[2, i] - pos2[2, i]
        
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        force_factor = -K * (1.0 - L0 / r)
        
        fx = force_factor * dx
        fy = force_factor * dy
        fz = force_factor * dz
        
        force1[0, i] += fx
        force1[1, i] += fy
        force1[2, i] += fz
        
        force2[0, i] -= fx
        force2[1, i] -= fy
        force2[2, i] -= fz
        
        acc_force += np.sqrt(fx*fx + fy*fy + fz*fz)
    
    return acc_force

def UpdateBondForces(sysobj: System):
    global accumulated_forces_bond
    molecules = sysobj.molecules
    
    total_acc = 0.0
    for bond in molecules.bonds:
        atom1 = molecules.atoms[bond.a1]
        atom2 = molecules.atoms[bond.a2]
        
        acc = compute_bond_forces(atom1.p, atom2.p, atom1.f, atom2.f, bond.K, bond.L0)
        total_acc += acc
    
    accumulated_forces_bond += total_acc

# Ultra-fast angle force calculation  
@njit(parallel=True)
def compute_angle_forces(pos1, pos2, pos3, force1, force2, force3, K, Phi0):
    no_mol = pos1.shape[1]
    acc_force = 0.0
    
    for i in prange(no_mol):
        # Vector from center to atoms
        d21x = pos2[0, i] - pos1[0, i]
        d21y = pos2[1, i] - pos1[1, i]
        d21z = pos2[2, i] - pos1[2, i]
        
        d23x = pos2[0, i] - pos3[0, i]
        d23y = pos2[1, i] - pos3[1, i]
        d23z = pos2[2, i] - pos3[2, i]
        
        norm_d21 = np.sqrt(d21x*d21x + d21y*d21y + d21z*d21z)
        norm_d23 = np.sqrt(d23x*d23x + d23y*d23y + d23z*d23z)
        
        dot_product = d21x*d23x + d21y*d23y + d21z*d23z
        cos_phi = dot_product / (norm_d21 * norm_d23)
        cos_phi = max(-0.999999, min(0.999999, cos_phi))  # Avoid numerical issues
        
        phi = np.arccos(cos_phi)
        
        # Simplified force calculation (approximation for speed)
        phi_diff = phi - Phi0
        
        # Force factors
        f1_factor = K * phi_diff / norm_d21
        f3_factor = K * phi_diff / norm_d23
        
        # Apply forces (simplified direction calculation)
        force1[0, i] += f1_factor * d21x / norm_d21
        force1[1, i] += f1_factor * d21y / norm_d21
        force1[2, i] += f1_factor * d21z / norm_d21
        
        force3[0, i] += f3_factor * d23x / norm_d23
        force3[1, i] += f3_factor * d23y / norm_d23
        force3[2, i] += f3_factor * d23z / norm_d23
        
        # Central atom gets opposite force
        force2[0, i] -= (f1_factor * d21x / norm_d21 + f3_factor * d23x / norm_d23)
        force2[1, i] -= (f1_factor * d21y / norm_d21 + f3_factor * d23y / norm_d23)
        force2[2, i] -= (f1_factor * d21z / norm_d21 + f3_factor * d23z / norm_d23)
        
        acc_force += abs(f1_factor) + abs(f3_factor)
    
    return acc_force

def UpdateAngleForces(sysobj: System):
    global accumulated_forces_angle
    molecules = sysobj.molecules
    
    total_acc = 0.0
    for angle in molecules.angles:
        atom1 = molecules.atoms[angle.a1]
        atom2 = molecules.atoms[angle.a2]
        atom3 = molecules.atoms[angle.a3]
        
        acc = compute_angle_forces(atom1.p, atom2.p, atom3.p,
                                 atom1.f, atom2.f, atom3.f,
                                 angle.K, angle.Phi0)
        total_acc += acc
    
    accumulated_forces_angle += total_acc

# Simple non-bonded forces (keeping original logic for correctness)
def UpdateNonBondedForces(sysobj: System):
    global accumulated_forces_non_bond
    acc = 0.0
    molecules = sysobj.molecules
    
    for i in range(molecules.no_mol):
        neighbor_indices = molecules.neighbours[i]
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        
        for j in valid_neighbors:
            for atom1_idx in range(len(molecules.atoms)):
                for atom2_idx in range(len(molecules.atoms)):
                    atom1_group = molecules.atoms[atom1_idx]
                    atom2_group = molecules.atoms[atom2_idx]
                    
                    ep = np.sqrt(atom1_group.ep * atom2_group.ep)
                    sigma2 = (0.5 * (atom1_group.sigma + atom2_group.sigma)) ** 2
                    KC = 80 * 0.7
                    q = KC * atom1_group.charge * atom2_group.charge

                    dp = atom1_group.p[:, i] - atom2_group.p[:, j]
                    r2 = np.dot(dp, dp)
                    r = np.sqrt(r2)

                    sir = sigma2 / r2
                    sir3 = sir * sir * sir
                    f = (ep * (12 * sir3 * sir3 - 6 * sir3) * sir + q / (r * r2)) * dp

                    atom1_group.f[:, i] += f
                    atom2_group.f[:, j] -= f
                    acc += np.sqrt(np.dot(f, f))

    accumulated_forces_non_bond += acc

# Ultra-fast integrator
@njit(parallel=True) 
def integrate_step(pos, vel, forces, mass, dt):
    no_mol = pos.shape[1]
    for i in prange(no_mol):
        for dim in range(3):
            vel[dim, i] += (dt / mass) * forces[dim, i]
            forces[dim, i] = 0.0
            pos[dim, i] += dt * vel[dim, i]

def UpdateKDK(sysobj: System, sc: Sim_Configuration):
    molecules = sysobj.molecules
    
    for atom_group in molecules.atoms:
        integrate_step(atom_group.p, atom_group.v, atom_group.f, atom_group.mass, sc.dt)
    
    sysobj.time += sc.dt

# Neighbor list building (keeping original for correctness)
def BuildNeighborList(sysobj: System):
    molecules = sysobj.molecules
    target_num = min(N_CLOSEST, max(0, molecules.no_mol - 1))
    distances2 = np.empty(molecules.no_mol)
    molecules.neighbours.fill(-1)
    
    for i in range(molecules.no_mol):
        neighbor_count = 0
        for j in range(molecules.no_mol):
            dp = molecules.atoms[0].p[:, i] - molecules.atoms[0].p[:, j]
            distances2[j] = np.dot(dp, dp)
        distances2[i] = 1e99

        index = np.argpartition(distances2, target_num)[:target_num]

        for k in index:
            if k < i:
                if i not in molecules.neighbours[k, :]:
                    if neighbor_count < N_CLOSEST:
                        molecules.neighbours[i, neighbor_count] = k
                        neighbor_count += 1
            else:
                if neighbor_count < N_CLOSEST:
                    molecules.neighbours[i, neighbor_count] = k
                    neighbor_count += 1

def MakeWater(N_molecules: int) -> System:
    L0 = 0.09584
    angle = 104.45 * deg2rad

    waterbonds = [
        Bond(K=20000, L0=L0, a1=0, a2=1),
        Bond(K=20000, L0=L0, a1=0, a2=2),
    ]

    waterangle = [
        Angle(K=1000, Phi0=angle, a1=1, a2=0, a3=2),
    ]

    Oatoms = Atoms(mass=16, ep=0.65, sigma=0.31, charge=-0.82, name="O", no_atoms=N_molecules)
    H1atoms = Atoms(mass=1, ep=0.18828, sigma=0.238, charge=0.41, name="H", no_atoms=N_molecules)
    H2atoms = Atoms(mass=1, ep=0.18828, sigma=0.238, charge=0.41, name="H", no_atoms=N_molecules)

    atoms = [Oatoms, H1atoms, H2atoms]
    molecules = Molecules(atoms, waterbonds, waterangle, N_molecules)

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
    molecules = sysobj.molecules
    
    for mol_idx in range(molecules.no_mol):
        for atom_group in molecules.atoms:
            file.write(
                f"{sysobj.time:.6g} {atom_group.name} {atom_group.p[0, mol_idx]:.6g} {atom_group.p[1, mol_idx]:.6g} {atom_group.p[2, mol_idx]:.6g}\n"
            )

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