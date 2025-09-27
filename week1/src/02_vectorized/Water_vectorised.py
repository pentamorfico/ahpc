# Applied High Performance Computing 
# 
# Molecular Dynamics Simulation of Water Molecules 
# 
# Description: This program simulates flexible water molecules using a simple
#              classical model. Each water has two covalent bonds and one angle.
#              All non-bonded atoms interact through LJ potential. 
#              Verlet integrator is used. 
#
# Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen

import sys
import time
import numpy as np
from typing import List

deg2rad = np.pi / 180.0 # pi/180 for changing degs to radians

accumulated_forces_bond: float = 0.0 # Checksum: accumulated size of forces
accumulated_forces_angle: float = 0.0 # Checksum: accumulated size of forces
accumulated_forces_non_bond: float = 0.0 # Checksum: accumulated size of forces

# Number of closest neighbors to consider in neighbor list. 
N_CLOSEST: int = 8

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
# atom class
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
        self.neighbours: np.ndarray = np.zeros((no_mol, N_CLOSEST), dtype=int)  # indices of the neighbours
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

# Update neighbour list for each atom, allowing us to quickly loop through all relevant non-bonded forces. Given the
# short timesteps, it takes many steps to go from being e.g. 20th closest to 2nd closest; only needs infrequent updating
def BuildNeighborList(sysobj: System):

    # We want at most N_CLOSEST neighbors, but no more than number of molecules.
    molecules = sysobj.molecules
    target_num = min(N_CLOSEST, max(0, molecules.no_mol - 1))

    distances2 = np.empty(molecules.no_mol) # array of distances to other molecules

    # Clear neighbor lists
    molecules.neighbours.fill(-1)  # Initialize with -1 (invalid index)
    
    for i in range(molecules.no_mol):     # For each molecule, build the neighbour list
        neighbor_count = 0
        for j in range(molecules.no_mol):
            # Use oxygen atom (index 0) position for distance calculation
            dp = molecules.atoms[0].p[:, i] - molecules.atoms[0].p[:, j]
            distances2[j] = np.dot(dp, dp)  # squared distance
        distances2[i] = 1e99 # exclude own molecule from neighbour list

        # argpartition partition so that target_num indices gives the smallest distances. Ignore the rest
        index = np.argpartition(distances2, target_num)[:target_num]

        # Test if index already exists in the neighbour list of other molecule and if not insert it in neighbour list of molecule i
        for k in index:  # k: molecule nr of the jth close molecule to molecule i
            if k < i:  # neighbour list of molecule k has already been created
                # Check if i is already in k's neighbor list
                if i not in molecules.neighbours[k, :]:  # molecule i is not in neighbour list of molecule k
                    if neighbor_count < N_CLOSEST:
                        molecules.neighbours[i, neighbor_count] = k  # add molecule k to the neighbour list of molecule i
                        neighbor_count += 1
            else:
                if neighbor_count < N_CLOSEST:
                    molecules.neighbours[i, neighbor_count] = k  # add molecule k to the neighbour list of molecule i
                    neighbor_count += 1

# Given a bond, updates the force on all atoms correspondingly
def UpdateBondForces(sysobj: System):
    global accumulated_forces_bond
    molecules = sysobj.molecules
    
    # Process all bonds for all molecules simultaneously
    for bond in molecules.bonds:
        # Get atom groups for this bond
        atom1_group = molecules.atoms[bond.a1]  # e.g., Oxygen atoms
        atom2_group = molecules.atoms[bond.a2]  # e.g., Hydrogen atoms
        
        # Vectorized calculation for all molecules
        dp = atom1_group.p - atom2_group.p  # shape: (3, no_mol)
        r = np.sqrt(np.sum(dp * dp, axis=0))  # shape: (no_mol,)
        
        # Force calculation: f = -K * (1 - L0/r) * dp
        force_factor = -bond.K * (1.0 - bond.L0 / r)  # shape: (no_mol,)
        f = force_factor[np.newaxis, :] * dp  # shape: (3, no_mol)
        
        # Apply forces
        atom1_group.f += f
        atom2_group.f -= f
        
        # Update checksum
        accumulated_forces_bond += np.sum(np.sqrt(np.sum(f * f, axis=0)))

# Iterates over all angles in molecules and updates forces on atoms correpondingly
def UpdateAngleForces(sysobj: System):
    global accumulated_forces_angle
    molecules = sysobj.molecules
    
    # Process all angles for all molecules simultaneously  
    for angle in molecules.angles:
        #====  angle forces  (H--O---H bonds) U_angle = 0.5*k_a(phi-phi_0)^2
        # f_H1 = K(phi-ph0)/|H1O|*Ta
        # f_H2 = K(phi-ph0)/|H2O|*Tc
        # f_O  = - (f_H1 + f_H2)
        # Ta = norm(H1O x (H1O x H2O))
        # Tc = norm(H2O x (H2O x H1O))
        #=============================================================
        atom1_group = molecules.atoms[angle.a1]  # H1 atoms
        atom2_group = molecules.atoms[angle.a2]  # O atoms (center)
        atom3_group = molecules.atoms[angle.a3]  # H2 atoms

        d21 = atom2_group.p - atom1_group.p  # O - H1, shape: (3, no_mol)
        d23 = atom2_group.p - atom3_group.p  # O - H2, shape: (3, no_mol)

        # phi = d21 dot d23 / |d21| |d23|
        norm_d21 = np.sqrt(np.sum(d21 * d21, axis=0))  # shape: (no_mol,)
        norm_d23 = np.sqrt(np.sum(d23 * d23, axis=0))  # shape: (no_mol,)
        dot_product = np.sum(d21 * d23, axis=0)  # shape: (no_mol,)
        phi = np.arccos(dot_product / (norm_d21 * norm_d23))  # shape: (no_mol,)

        # Cross products for all molecules
        c21_23 = np.cross(d21.T, d23.T).T  # shape: (3, no_mol)
        Ta = np.cross(d21.T, c21_23.T).T  # shape: (3, no_mol)
        Ta_mag = np.sqrt(np.sum(Ta * Ta, axis=0))  # shape: (no_mol,)
        Ta = Ta / Ta_mag[np.newaxis, :]  # normalize

        Tc = np.cross(c21_23.T, d23.T).T  # shape: (3, no_mol)
        Tc_mag = np.sqrt(np.sum(Tc * Tc, axis=0))  # shape: (no_mol,)
        Tc = Tc / Tc_mag[np.newaxis, :]  # normalize

        # Force calculations
        force_factor1 = angle.K * (phi - angle.Phi0) / norm_d21  # shape: (no_mol,)
        force_factor3 = angle.K * (phi - angle.Phi0) / norm_d23  # shape: (no_mol,)
        
        f1 = Ta * force_factor1[np.newaxis, :]  # shape: (3, no_mol)
        f3 = Tc * force_factor3[np.newaxis, :]  # shape: (3, no_mol)

        # Apply forces
        atom1_group.f += f1
        atom2_group.f -= (f1 + f3)
        atom3_group.f += f3

        # Update checksum
        accumulated_forces_angle += np.sum(np.sqrt(np.sum(f1 * f1, axis=0))) + np.sum(np.sqrt(np.sum(f3 * f3, axis=0)))

# Iterates over atoms in different molecules and calculate non-bonded forces
def UpdateNonBondedForces(sysobj: System):
    """
    nonbonded forces: only a force between atoms in different molecules
    The total non-bonded forces come from Lennard Jones (LJ) and coulomb interactions
    U = ep[(sigma/r)^12-(sigma/r)^6] + C*q1*q2/r
    """
    global accumulated_forces_non_bond
    acc = 0.0
    molecules = sysobj.molecules
    
    for i in range(molecules.no_mol):
        # Get valid neighbors for molecule i (neighbors are stored with -1 for invalid entries)
        neighbor_indices = molecules.neighbours[i]
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        
        for j in valid_neighbors:  # iterate over all neighbours of molecule i
            for atom1_idx in range(len(molecules.atoms)):  # atom types in molecule i
                for atom2_idx in range(len(molecules.atoms)):  # atom types in molecule j
                    atom1_group = molecules.atoms[atom1_idx]
                    atom2_group = molecules.atoms[atom2_idx]
                    
                    ep = np.sqrt(atom1_group.ep * atom2_group.ep)  # ep = sqrt(ep1*ep2)
                    sigma2 = (0.5 * (atom1_group.sigma + atom2_group.sigma)) ** 2  # sigma = (sigma1+sigma2)/2
                    KC = 80 * 0.7  # Coulomb prefactor
                    q = KC * atom1_group.charge * atom2_group.charge

                    dp = atom1_group.p[:, i] - atom2_group.p[:, j]  # shape: (3,)
                    r2 = np.dot(dp, dp)  # scalar
                    r = np.sqrt(r2)

                    sir = sigma2 / r2  # crossection**2 times inverse squared distance
                    sir3 = sir * sir * sir
                    # LJ + Coulomb forces
                    f = (ep * (12 * sir3 * sir3 - 6 * sir3) * sir + q / (r * r2)) * dp

                    atom1_group.f[:, i] += f
                    # update both pairs, since the force is equal and opposite and pairs only exist in one neighbor list
                    atom2_group.f[:, j] -= f
                    acc += np.sqrt(np.dot(f, f))

    accumulated_forces_non_bond += acc


# integrating the system for one time step using Leapfrog symplectic integration
def UpdateKDK(sysobj: System, sc: Sim_Configuration):
    molecules = sysobj.molecules
    
    for atom_group in molecules.atoms:
        # Update velocities: v += (dt/mass) * f
        atom_group.v += (sc.dt / atom_group.mass) * atom_group.f
        
        # Clear forces for next step
        atom_group.f.fill(0.0)
        
        # Update positions: p += dt * v
        atom_group.p += sc.dt * atom_group.v

    sysobj.time += sc.dt  # update time


# Setup one water molecule
def MakeWater(N_molecules: int) -> System:
    #===========================================================
    # creating water molecules at position X0,Y0,Z0. 3 atoms
    #                        H---O---H
    # The angle is 104.45 degrees and bond length is 0.09584 nm
    #===========================================================
    # mass units of dalton
    # initial velocity and force is set to zero for all the atoms by the constructor
    L0 = 0.09584
    angle = 104.45 * deg2rad

    # bonds beetween first H-O and second H-O respectively
    waterbonds = [
        Bond(K=20000, L0=L0, a1=0, a2=1),
        Bond(K=20000, L0=L0, a1=0, a2=2),
    ]

    # angle between H-O-H
    waterangle = [
        Angle(K=1000, Phi0=angle, a1=1, a2=0, a3=2),
    ]

    # Create SoA structure for all water molecules
    # Create Atoms objects for O, H1, H2 atoms across all molecules
    Oatoms = Atoms(mass=16, ep=0.65, sigma=0.31, charge=-0.82, name="O", no_atoms=N_molecules)
    H1atoms = Atoms(mass=1, ep=0.18828, sigma=0.238, charge=0.41, name="H", no_atoms=N_molecules)
    H2atoms = Atoms(mass=1, ep=0.18828, sigma=0.238, charge=0.41, name="H", no_atoms=N_molecules)

    atoms = [Oatoms, H1atoms, H2atoms]
    molecules = Molecules(atoms, waterbonds, waterangle, N_molecules)

    # initialize all water molecules on a sphere.
    phi = np.arccos(-1) * (np.sqrt(5.0) - 1.0)
    radius = np.sqrt(N_molecules) * 0.15

    for i in range(N_molecules):
        y = 1.0 - (i / (N_molecules - 1.0)) if N_molecules > 1 else 0.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i

        x = np.cos(theta) * r
        z = np.sin(theta) * r

        P0 = Vec3(x * radius, y * radius, z * radius)
        
        # Set positions in SoA structure
        Oatoms.p[:, i] = [P0[0], P0[1], P0[2]]
        H1atoms.p[:, i] = [P0[0] + L0 * np.sin(angle / 2), P0[1] + L0 * np.cos(angle / 2), P0[2]]
        H2atoms.p[:, i] = [P0[0] - L0 * np.sin(angle / 2), P0[1] + L0 * np.cos(angle / 2), P0[2]]

    sysobj = System()
    sysobj.molecules = molecules

    return sysobj

# Write the system configurations in the trajectory file.
def WriteOutput(sysobj: System, file):
    # Loop over all atoms in model one molecule at a time and write out position
    molecules = sysobj.molecules
    
    for mol_idx in range(molecules.no_mol):
        for atom_group in molecules.atoms:
            # Match C++ iostream default formatting (~6 significant digits, general format)
            file.write(
                f"{sysobj.time:.6g} {atom_group.name} {atom_group.p[0, mol_idx]:.6g} {atom_group.p[1, mol_idx]:.6g} {atom_group.p[2, mol_idx]:.6g}\n"
            )

#======================================================================================================
#======================== Main function ===============================================================
#======================================================================================================
sc = Sim_Configuration(sys.argv) # Load the system configuration from command line data

sysobj = MakeWater(sc.no_mol) # this will create a system containing sc.no_mol water molecules

file = open(sc.filename, "w") # open file

WriteOutput(sysobj, file) # writing the initial configuration in the trajectory file

tstart = time.perf_counter() # start time (seconds)

for step in range(sc.steps):
    # BuildNeighborList every 100th step
    if step % 100 == 0:
        BuildNeighborList(sysobj)

    # Always evolve the system
    UpdateBondForces(sysobj)
    UpdateAngleForces(sysobj)
    UpdateNonBondedForces(sysobj)
    UpdateKDK(sysobj, sc)

    # Write output every data_period steps
    if step % sc.data_period == 0:
        WriteOutput(sysobj, file)
    
file.close()

tend = time.perf_counter() # end time (seconds)

# Print summaries
print(f"Accumulated forces Bonds   : {accumulated_forces_bond:9.5f}")
print(f"Accumulated forces Angles  : {accumulated_forces_angle:9.5f}")
print(f"Accumulated forces Non-bond: {accumulated_forces_non_bond:9.5f}")
print(f"Elapsed total time:       {tend - tstart:9.4f}")