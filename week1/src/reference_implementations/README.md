# Reference Implementations

This folder contains the original C++ and Fortran implementations provided with the assignment.

## Structure

- `cpp/` - C++ implementations (sequential and vectorized)
- `fortran/` - Fortran implementations (sequential and vectorized)

## Files

### C++ Implementations
- `Water_sequential.cpp` - Sequential C++ implementation
- `Water_vectorised.cpp` - Vectorized C++ implementation  
- `Makefile` - Build configuration for C++ versions

### Fortran Implementations
- `Water_sequential.f90` - Sequential Fortran implementation
- `Water_vectorised.f90` - Vectorized Fortran implementation
- `Makefile` - Build configuration for Fortran versions

## Usage

### C++
```bash
cd cpp
make
./water_sequential
./water_vectorised
```

### Fortran
```bash
cd fortran
make
./water_sequential
./water_vectorised
```

## Purpose

These implementations serve as:
- Reference implementations for comparison
- Examples of traditional HPC languages
- Baseline performance targets for Python optimizations
- Educational examples of C++/Fortran optimization techniques

The Python implementations in this project demonstrate that modern Python can achieve competitive performance with these traditional HPC languages through appropriate optimization techniques.