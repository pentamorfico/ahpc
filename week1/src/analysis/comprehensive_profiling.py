#!/usr/bin/env python3
"""
COMPREHENSIVE PROFILING ANALYSIS
=================================

Profile ALL implementations to identify exact bottlenecks:
1. Sequential (original)
2. Vectorized (NumPy) 
3. Pure Arrays + Numba
4. GPU (CuPy fallback)

Uses cProfile and line_profiler for detailed analysis
"""

import cProfile
import pstats
import io
import time
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add path for imports
import sys
import warnings
warnings.filterwarnings("ignore")

class ProfiledSimulation:
    """Wrapper for clean profiling of different implementations"""
    
    def __init__(self, name, color="🔸"):
        self.name = name
        self.color = color
        self.profile_data = {}
    
    def profile_function(self, func, *args, **kwargs):
        """Profile a specific function with cProfile"""
        profiler = cProfile.Profile()
        
        # Time the execution
        start_time = time.perf_counter()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        end_time = time.perf_counter()
        
        # Capture profile stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        self.profile_data = {
            'total_time': end_time - start_time,
            'result': result,
            'profile_text': s.getvalue()
        }
        
        return result
    
    def print_analysis(self):
        """Print detailed profiling analysis"""
        print(f"\n{self.color} {self.name.upper()} PROFILING ANALYSIS")
        print("=" * 60)
        print(f"⏱️  Total Time: {self.profile_data['total_time']:.4f} seconds")
        print(f"📊 Top Function Calls:")
        print(self.profile_data['profile_text'][:1000] + "..." if len(self.profile_data['profile_text']) > 1000 else self.profile_data['profile_text'])

# ===================================================================
# IMPLEMENTATION 1: SEQUENTIAL (problematic original avoided)
# ===================================================================

def create_sequential_system(n_molecules):
    """Create system mimicking sequential approach"""
    class Atom:
        def __init__(self, mass, charge, name):
            self.mass = mass
            self.charge = charge  
            self.name = name
            self.p = np.array([0.0, 0.0, 0.0])
            self.v = np.array([0.0, 0.0, 0.0])
            self.f = np.array([0.0, 0.0, 0.0])
    
    class Molecule:
        def __init__(self):
            self.atoms = []
            self.neighbours = []
    
    class System:
        def __init__(self):
            self.molecules = []
    
    system = System()
    
    for mol in range(n_molecules):
        molecule = Molecule()
        
        # Create water atoms
        o_atom = Atom(15.999, -0.8476, "O")
        h1_atom = Atom(1.008, 0.4238, "H")  
        h2_atom = Atom(1.008, 0.4238, "H")
        
        # Set positions
        o_atom.p = np.array([mol * 3.0, 0.0, 0.0])
        h1_atom.p = np.array([mol * 3.0 + 0.9584, 0.0, 0.0])
        h2_atom.p = np.array([mol * 3.0 + 0.48, 0.83, 0.0])
        
        # Add noise
        o_atom.p += np.random.normal(0, 0.1, 3)
        h1_atom.p += np.random.normal(0, 0.1, 3)
        h2_atom.p += np.random.normal(0, 0.1, 3)
        
        molecule.atoms = [o_atom, h1_atom, h2_atom]
        system.molecules.append(molecule)
    
    # Simple neighbor list
    for i in range(n_molecules):
        for j in range(n_molecules):
            if i != j:
                system.molecules[i].neighbours.append(j)
    
    return system

def run_sequential_style(system, n_steps, dt):
    """Simulate sequential-style computation"""
    for step in range(n_steps):
        # Bond forces (simplified)
        for mol in system.molecules:
            if len(mol.atoms) >= 3:
                # O-H bonds
                for i in [0, 0]:  # O-H1, O-H2
                    for j in [1, 2]:
                        if i != j:
                            atom1, atom2 = mol.atoms[i], mol.atoms[j]
                            dr = atom2.p - atom1.p
                            r = np.linalg.norm(dr)
                            if r > 0:
                                force = -450.0 * (r - 0.9584) * dr / r
                                atom1.f -= force
                                atom2.f += force
        
        # Non-bonded forces
        for i, mol_i in enumerate(system.molecules):
            for j in mol_i.neighbours[:min(10, len(mol_i.neighbours))]:  # Limit for speed
                if j < len(system.molecules):
                    mol_j = system.molecules[j]
                    for atom1 in mol_i.atoms:
                        for atom2 in mol_j.atoms:
                            dr = atom2.p - atom1.p
                            r2 = np.dot(dr, dr)
                            if r2 > 0.25 and r2 < 100:
                                r = np.sqrt(r2)
                                # Simplified force
                                force = 0.1 * dr / r**3
                                atom1.f += force
                                atom2.f -= force
        
        # Integration
        for mol in system.molecules:
            for atom in mol.atoms:
                if atom.mass > 0:
                    acc = atom.f / atom.mass
                    atom.v += acc * dt * 0.5
                    atom.p += atom.v * dt
                    atom.f.fill(0.0)

# ===================================================================
# IMPLEMENTATION 2: VECTORIZED (NumPy-style)
# ===================================================================

def create_vectorized_system(n_molecules):
    """Create vectorized system with NumPy arrays"""
    n_atoms = n_molecules * 3
    
    return {
        'positions': np.random.normal(0, 1, (n_atoms, 3)),
        'velocities': np.zeros((n_atoms, 3)),
        'forces': np.zeros((n_atoms, 3)),
        'masses': np.tile([15.999, 1.008, 1.008], n_molecules),
        'charges': np.tile([-0.8476, 0.4238, 0.4238], n_molecules)
    }

def run_vectorized_style(system, n_steps, dt):
    """Simulate vectorized computation"""
    n_atoms = system['positions'].shape[0]
    n_molecules = n_atoms // 3
    
    for step in range(n_steps):
        system['forces'].fill(0.0)
        
        # Vectorized bond forces
        for mol in range(n_molecules):
            base = mol * 3
            o, h1, h2 = base, base + 1, base + 2
            
            # O-H1 bond
            dr = system['positions'][h1] - system['positions'][o]
            r = np.linalg.norm(dr)
            if r > 0:
                force = -450.0 * (r - 0.9584) * dr / r
                system['forces'][o] -= force
                system['forces'][h1] += force
            
            # O-H2 bond  
            dr = system['positions'][h2] - system['positions'][o]
            r = np.linalg.norm(dr)
            if r > 0:
                force = -450.0 * (r - 0.9584) * dr / r
                system['forces'][o] -= force
                system['forces'][h2] += force
        
        # Vectorized non-bonded (simplified)
        for i in range(0, n_atoms, 3):  # Every 3rd atom (oxygen only)
            for j in range(i + 3, min(i + 30, n_atoms), 3):  # Limited range
                dr = system['positions'][j] - system['positions'][i]
                r2 = np.dot(dr, dr)
                if r2 > 0.25 and r2 < 100:
                    r = np.sqrt(r2)
                    force = 0.1 * dr / r**3
                    system['forces'][i] += force
                    system['forces'][j] -= force
        
        # Vectorized integration
        acc = system['forces'] / system['masses'][:, np.newaxis]
        system['velocities'] += acc * dt * 0.5
        system['positions'] += system['velocities'] * dt

# ===================================================================
# IMPLEMENTATION 3: PURE ARRAYS + NUMBA
# ===================================================================

def run_numba_simulation(n_molecules, n_steps, dt):
    """Use our optimized numba implementation"""
    from clean_optimized import run_clean_simulation
    return run_clean_simulation(n_molecules, n_steps, dt)

# ===================================================================
# IMPLEMENTATION 4: GPU (CPU fallback)
# ===================================================================

def run_gpu_simulation(n_molecules, n_steps, dt):
    """GPU-style computation (CPU fallback)"""
    # Use NumPy as CuPy fallback
    n_atoms = n_molecules * 3
    
    positions = np.random.normal(0, 1, (n_atoms, 3)).astype(np.float32)
    velocities = np.zeros((n_atoms, 3), dtype=np.float32)
    forces = np.zeros((n_atoms, 3), dtype=np.float32)
    masses = np.tile([15.999, 1.008, 1.008], n_molecules).astype(np.float32)
    
    for step in range(n_steps):
        forces.fill(0.0)
        
        # Simplified GPU-style computation
        # All operations use float32 for GPU similarity
        for i in range(n_atoms):
            for j in range(i + 1, min(i + 50, n_atoms)):  # Limited for speed
                if (i // 3) == (j // 3):  # Skip same molecule
                    continue
                    
                dr = positions[j] - positions[i]
                r2 = np.sum(dr * dr)
                if r2 > 0.25 and r2 < 100:
                    r_inv = 1.0 / np.sqrt(r2)
                    force = 0.1 * dr * r_inv**3
                    forces[i] += force
                    forces[j] -= force
        
        # Integration
        acc = forces / masses[:, np.newaxis]
        velocities += acc * dt * 0.5
        positions += velocities * dt

# ===================================================================
# PROFILING ORCHESTRATOR
# ===================================================================

def run_comprehensive_profiling():
    """Profile all implementations and compare bottlenecks"""
    
    print("🔬 COMPREHENSIVE PROFILING ANALYSIS")
    print("=" * 80)
    print("Profiling all MD implementations to identify bottlenecks...")
    print()
    
    n_molecules = 200  # Manageable size for profiling
    n_steps = 50      # Fewer steps for detailed analysis
    dt = 0.001
    
    implementations = []
    
    # 1. Sequential-style profiling
    print("1️⃣ Profiling Sequential-style implementation...")
    seq_profiler = ProfiledSimulation("Sequential", "🔴")
    seq_system = create_sequential_system(n_molecules)
    seq_profiler.profile_function(run_sequential_style, seq_system, n_steps, dt)
    implementations.append(seq_profiler)
    
    # 2. Vectorized profiling
    print("2️⃣ Profiling Vectorized NumPy implementation...")
    vec_profiler = ProfiledSimulation("Vectorized", "🔵")
    vec_system = create_vectorized_system(n_molecules)
    vec_profiler.profile_function(run_vectorized_style, vec_system, n_steps, dt)
    implementations.append(vec_profiler)
    
    # 3. Pure Arrays + Numba profiling
    print("3️⃣ Profiling Pure Arrays + Numba implementation...")
    numba_profiler = ProfiledSimulation("Pure Numba", "🟢")
    numba_profiler.profile_function(run_numba_simulation, n_molecules, n_steps, dt)
    implementations.append(numba_profiler)
    
    # 4. GPU-style profiling
    print("4️⃣ Profiling GPU-style implementation...")
    gpu_profiler = ProfiledSimulation("GPU-style", "🟡")  
    gpu_profiler.profile_function(run_gpu_simulation, n_molecules, n_steps, dt)
    implementations.append(gpu_profiler)
    
    # Print detailed analysis for each
    for impl in implementations:
        impl.print_analysis()
    
    # Comparative summary
    print(f"\n🏆 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Implementation':<15} | {'Time (s)':<8} | {'Relative':<8} | {'Status':<10}")
    print("-" * 60)
    
    times = [impl.profile_data['total_time'] for impl in implementations]
    fastest_time = min(times)
    
    for impl in implementations:
        time_taken = impl.profile_data['total_time']
        relative = time_taken / fastest_time
        
        if relative == 1.0:
            status = "🏆 FASTEST"
        elif relative < 2.0:
            status = "✅ Good"
        elif relative < 5.0:
            status = "⚠️ Slow"
        else:
            status = "🐌 Very Slow"
            
        print(f"{impl.name:<15} | {time_taken:<8.3f} | {relative:<8.1f}x | {status:<10}")
    
    print(f"\n💡 KEY INSIGHTS:")
    fastest_impl = implementations[times.index(fastest_time)]
    print(f"   • {fastest_impl.name} is the fastest implementation")
    print(f"   • Profiling shows exactly where each approach spends time")
    print(f"   • Pure array + Numba should dominate due to compiled code")
    print(f"   • Object-oriented approaches suffer from method call overhead")

if __name__ == "__main__":
    run_comprehensive_profiling()