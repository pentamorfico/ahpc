# AHPC Assignment 1: Molecular Dynamics Simulation of Water Molecules

## Overview

This project implements a molecular dynamics (MD) simulation of N water molecules using different programming approaches (C++, Fortran, and Python) with focus on vectorization and performance optimization.

### Assignment Details
- **Deadline**: Saturday 29/9 23:59
- **Report**: Up to 3 pages (excluding code)
- **Deliverables**: Vectorized (Struct-of-Array) code as source file and PDF

## Background

N-body simulations are fundamental in physics and chemistry with O(N²) complexity. This implementation focuses on molecular dynamics where atoms interact through energy potentials, and system evolution is obtained by numerically integrating Newton's equations of motion.

### Key Approximations
- Only considers 8 nearest neighbor molecules for intermolecular forces
- Neighbor lists updated every 100 steps
- Uses leap-frog integrator for time evolution

## Physics Model

### Intramolecular Potentials

1. **Harmonic Bond Potential**:
   ```
   V_bond(r) = (k_b/2)(r - l_0)²
   ```
   - k_b: force constant
   - l_0 = 95.48 pm: bond length

2. **Angle Potential**:
   ```
   V_angle(φ) = (k_a/2)(φ - φ_0)²
   ```
   - k_a: force constant  
   - φ_0 = 104.45°: bond angle

### Intermolecular Potentials

1. **Lennard-Jones Potential**:
   ```
   V_LJ(r) = ε_ij[(σ_ij/r)¹² - (σ_ij/r)⁶]
   ```
   - ε_ij = √(ε_i × ε_j)
   - σ_ij = (σ_i + σ_j)/2

2. **Coulomb Potential**:
   ```
   V_E(r) = k(q₁q₂)/r
   ```

### Time Integration

**Leap-frog Integrator**:
```
v(t + Δt/2) = v(t - Δt/2) + (Δt/m)F(t)
r(t + Δt) = r(t) + Δt × v(t + Δt/2)
```

## Project Structure

```
ahpc/
├── week1/
│   ├── 6water.avi                 # Example visualization
│   ├── Water_visualizer.ipynb     # Visualization notebook
│   ├── cpp/
│   │   ├── Makefile
│   │   ├── Water_sequential.cpp   # Reference sequential version
│   │   └── Water_vectorised.cpp   # Vectorized implementation
│   ├── fortran/
│   │   ├── Makefile
│   │   ├── Water_sequential.f90
│   │   └── Water_vectorised.f90
│   └── python/
│       ├── Water_sequential_lists.py
│       ├── Water_sequential.py     # Reference with NumPy
│       └── Water_vectorised.py     # Vectorized implementation
```

## Getting Started

### Prerequisites
- C++ compiler (g++)
- Fortran compiler (gfortran)
- Python with NumPy
- Access to DAG/ERDA environment

### Setup
```bash
mkdir AHPC
cd AHPC
git clone https://github.com/haugboel/ahpc.git
```

### Compilation (C++/Fortran)
```bash
cd ahpc/week1/cpp  # or fortran
make
```

This produces two binaries:
- `seq`: Sequential reference implementation
- `vec`: Vectorized implementation

### Running Simulations
```bash
./seq -steps <number> -no_mol <molecules> -fwrite <frequency> -dt <timestep> -ofile <filename>
```

Example:
```bash
./seq -steps 10000 -no_mol 100 -fwrite 100
```

### Command Line Options
- `-steps`: Number of simulation steps
- `-no_mol`: Number of molecules  
- `-fwrite`: I/O frequency
- `-dt`: Timestep size
- `-ofile`: Output filename

## Tasks

### Task 1: Struct-of-Arrays (SoA) Implementation

**Objective**: Convert Array-of-Structures (AoS) to Struct-of-Arrays (SoA) layout for better vectorization.

**Changes Required**:
- Replace `std::vector<Molecule>` with `Molecules` class
- Restructure loops to iterate over identical atoms
- Modify initialization, output, and integrator functions

**Before (AoS)**:
```cpp
for (Molecule& molecule : sys.molecules)
    for (auto& atom : molecule.atoms) {
        // process atom
    }
```

**After (SoA)**:
```cpp
Molecules& molecule = sys.molecules;
for (auto& atom : molecule.atoms)
    for (int i = 0; i < molecule.no_mol; i++) {
        // process atom[i]
    }
```

**Verification**: Both `seq` and `vec` executables should produce identical checksums.

### Task 2: Performance Profiling

#### C++/Fortran Profiling with gprof
1. Compile with profiling flags: `-pg`
2. Run simulation: `./seq -steps 100000 -no_mol 4`
3. Generate profile: `gprof -p -b ./seq gmon.out`

#### Python Profiling
1. **Function-level**: `python -m cProfile Water_sequential.py > out.prof`
2. **Line-level**: `python -m kernprof -lv -p Water_sequential.py`

#### Analysis Questions
1. Which functions contribute most to runtime (100 molecules)?
2. How does performance scale with system size (10, 100, 1000, 50000 molecules)?
3. Which functions are most critical for optimization?
4. Performance comparison: SoA vs AoS versions

## Code Architecture

### Class Structure
- **System**: Contains molecules and global time
- **Molecule/Molecules**: Contains atoms, bonds, angles, neighbors
- **Atom/Atoms**: Positions, velocities, forces, charges, LJ parameters
- **Bond/Angle**: Force constants and equilibrium values
- **Neighbors**: Lists of up to 8 nearest neighbors

### Key Functions
- `UpdateBondForces()`: Calculates harmonic bond forces
- `UpdateAngleForces()`: Calculates angle forces  
- `UpdateNonBondedForces()`: Calculates LJ and Coulomb forces
- `BuildNeighborList()`: Updates neighbor lists (every 100 steps)
- `UpdateKDK()`: Leap-frog time integration

## Visualization

Use the provided Jupyter notebook to visualize simulation results:
1. Set `datadir` variable to trajectory file location
2. Run visualization cells
3. View 3D molecular dynamics evolution

The simulation shows water molecules initially arranged on a sphere, collapsing inward like an air bubble, with droplets being ejected.

## Performance Optimization Tips

1. **Compilation Flags**: Use `-O3 -ffast-math` for optimal performance
2. **Memory Layout**: SoA improves vectorization and cache efficiency  
3. **Neighbor Lists**: Reduces O(N²) complexity to approximately O(N)
4. **SIMD**: Modern compilers can auto-vectorize SoA loops

## Expected Output

Simulation produces checksums for verification:
```
Accumulated forces Bonds    : 3.4061e+08
Accumulated forces Angles   : 2.6864e+08  
Accumulated forces Non-bond : 5.7982e+09
Elapsed total time          : 1.7036
```

## Resources

- **ERDA**: File storage and computing environment
- **DAG**: Testing, visualization, and benchmarks
- **Jupyter**: Interactive development and visualization

## Report Requirements

Include in your report:
- Verification of correct checksums
- Brief description of code modifications (1-3 lines per function)
- Profiling results and performance analysis
- Comparison of AoS vs SoA implementations
- Discussion of scalability and optimization opportunities