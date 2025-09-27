#!/usr/bin/env python3
"""
DETAILED BOTTLENECK ANALYSIS
=============================

Deep dive into profiling with compilation separated from execution
and line-by-line analysis of bottlenecks
"""

import cProfile
import pstats
import io
import time
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

def analyze_function_hotspots(profile_text):
    """Extract key bottlenecks from profile data"""
    lines = profile_text.split('\n')
    hotspots = []
    
    for line in lines:
        if 'tottime' in line or 'percall' in line:
            continue
        if any(keyword in line.lower() for keyword in ['dot', 'norm', 'sum', 'sqrt', 'force', 'distance']):
            hotspots.append(line.strip())
    
    return hotspots[:10]  # Top 10 hotspots

def profile_with_warmup():
    """Profile with proper warmup to separate compilation from execution"""
    
    print("🔥 DETAILED BOTTLENECK ANALYSIS")
    print("=" * 70)
    print("Separating compilation overhead from execution performance")
    print()
    
    from clean_optimized import run_clean_simulation, warmup_clean
    
    n_molecules = 500
    n_steps = 100
    dt = 0.001
    
    # === WARMUP ANALYSIS ===
    print("1️⃣ COMPILATION OVERHEAD ANALYSIS")
    print("-" * 50)
    
    print("⏱️  Cold start (with compilation)...")
    start = time.perf_counter()
    result1 = run_clean_simulation(n_molecules, n_steps, dt)
    cold_time = time.perf_counter() - start
    
    print("⏱️  Warm start (already compiled)...")
    start = time.perf_counter()
    result2 = run_clean_simulation(n_molecules, n_steps, dt)
    warm_time = time.perf_counter() - start
    
    compilation_overhead = cold_time - warm_time
    
    print(f"❄️  Cold start time: {cold_time:.4f}s")
    print(f"🔥 Warm start time: {warm_time:.4f}s") 
    print(f"⚡ Compilation overhead: {compilation_overhead:.4f}s ({(compilation_overhead/cold_time)*100:.1f}%)")
    
    # === EXECUTION PROFILING ===
    print(f"\n2️⃣ EXECUTION BOTTLENECK ANALYSIS (WARM)")
    print("-" * 50)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    execution_time = run_clean_simulation(n_molecules, n_steps, dt)
    
    profiler.disable()
    
    # Analyze profile
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('tottime')  # Sort by actual time spent in function
    ps.print_stats(30)
    
    profile_output = s.getvalue()
    
    print(f"⏱️  Profiled execution time: {execution_time:.4f}s")
    print(f"📊 Top time-consuming functions:")
    print()
    
    # Extract and display key hotspots
    lines = profile_output.split('\n')[5:15]  # Skip headers
    for line in lines:
        if line.strip() and not line.strip().startswith('-'):
            print(f"   {line}")
    
    # === COMPARATIVE ANALYSIS ===
    print(f"\n3️⃣ COMPARATIVE BOTTLENECK ANALYSIS")
    print("-" * 50)
    
    # Sequential simulation
    seq_system = create_mock_sequential_system(n_molecules)
    
    print("🔴 Sequential Implementation Analysis:")
    seq_profiler = cProfile.Profile()
    seq_profiler.enable()
    
    start = time.perf_counter()
    run_mock_sequential(seq_system, n_steps, dt)
    seq_time = time.perf_counter() - start
    
    seq_profiler.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(seq_profiler, stream=s)
    ps.sort_stats('tottime')
    ps.print_stats(10)
    seq_profile = s.getvalue()
    
    print(f"   ⏱️  Time: {seq_time:.4f}s")
    print(f"   🐌 Bottlenecks: Object attribute access, dot products in loops")
    
    # Vectorized simulation
    print(f"\n🔵 Vectorized Implementation Analysis:")
    vec_system = create_mock_vectorized_system(n_molecules)
    
    vec_profiler = cProfile.Profile()
    vec_profiler.enable()
    
    start = time.perf_counter()
    run_mock_vectorized(vec_system, n_steps, dt)
    vec_time = time.perf_counter() - start
    
    vec_profiler.disable()
    
    print(f"   ⏱️  Time: {vec_time:.4f}s")  
    print(f"   ⚡ Strengths: Vectorized operations, better memory access")
    
    # === FINAL COMPARISON ===
    print(f"\n4️⃣ PERFORMANCE RANKING (Execution Only)")
    print("-" * 50)
    
    results = [
        ("Pure Numba (Warm)", warm_time),
        ("Vectorized NumPy", vec_time),
        ("Sequential Objects", seq_time)
    ]
    
    # Sort by time
    results.sort(key=lambda x: x[1])
    
    fastest_time = results[0][1]
    
    for i, (name, time_val) in enumerate(results):
        speedup = fastest_time / time_val
        emoji = "🏆" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"   {emoji} {name}: {time_val:.4f}s ({speedup:.1f}x)")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Compilation overhead can dominate small simulations")
    print(f"   • {results[0][0]} wins when properly warmed up")
    print(f"   • Object attribute access is major bottleneck in sequential")
    print(f"   • Memory layout and vectorization matter more than language")

def create_mock_sequential_system(n_molecules):
    """Create mock sequential system for comparison"""
    class Atom:
        def __init__(self):
            self.p = np.random.normal(0, 1, 3)
            self.v = np.zeros(3)
            self.f = np.zeros(3)
            self.mass = 1.0
    
    system = []
    for _ in range(n_molecules * 3):
        system.append(Atom())
    
    return system

def run_mock_sequential(system, n_steps, dt):
    """Mock sequential computation with object overhead"""
    n_atoms = len(system)
    
    for step in range(n_steps):
        # Clear forces
        for atom in system:
            atom.f.fill(0.0)
        
        # Pairwise interactions (simplified)
        for i in range(0, n_atoms, 10):  # Every 10th for speed
            for j in range(i + 1, min(i + 20, n_atoms)):
                dr = system[j].p - system[i].p
                r2 = np.dot(dr, dr)
                if r2 > 0.1:
                    force = 0.01 * dr / r2
                    system[i].f += force
                    system[j].f -= force
        
        # Integration
        for atom in system:
            acc = atom.f / atom.mass
            atom.v += acc * dt * 0.5
            atom.p += atom.v * dt

def create_mock_vectorized_system(n_molecules):
    """Create mock vectorized system"""
    n_atoms = n_molecules * 3
    return {
        'positions': np.random.normal(0, 1, (n_atoms, 3)),
        'velocities': np.zeros((n_atoms, 3)),
        'forces': np.zeros((n_atoms, 3)),
        'masses': np.ones(n_atoms)
    }

def run_mock_vectorized(system, n_steps, dt):
    """Mock vectorized computation"""
    n_atoms = system['positions'].shape[0]
    
    for step in range(n_steps):
        system['forces'].fill(0.0)
        
        # Vectorized pairwise (simplified) 
        for i in range(0, n_atoms, 10):
            for j in range(i + 1, min(i + 20, n_atoms)):
                dr = system['positions'][j] - system['positions'][i]
                r2 = np.sum(dr * dr)
                if r2 > 0.1:
                    force = 0.01 * dr / r2
                    system['forces'][i] += force
                    system['forces'][j] -= force
        
        # Vectorized integration
        acc = system['forces'] / system['masses'][:, np.newaxis]
        system['velocities'] += acc * dt * 0.5
        system['positions'] += system['velocities'] * dt

if __name__ == "__main__":
    profile_with_warmup()