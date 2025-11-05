# instruct h5fc to use mpic++ as the compiler and linker command
export HDF5_FC=mpif90
export HDF5_FLINKER=mpif90

# Fastest executable (-ffast-math removes checking for NaNs and other things)
OPT=-O3

# Debug flags with runtime checks (slow!)
DBG = -fbacktrace -ffpe-trap=invalid,zero,overflow -fcheck=all
DBG =

FFLAGS := $(OPT) $(DBG) -Wall -march=native -cpp

.PHONY: clean all

all: fwc_sequential fwc_parallel

fwc_sequential: fwc_sequential.f90
	h5fc $(FFLAGS) -o fwc_sequential fwc_sequential.f90

fwc_parallel: fwc_parallel.f90 fwc_sequential
	h5fc $(FFLAGS) -o fwc_parallel fwc_parallel.f90

clean:
	rm -f fwc_sequential fwc_parallel *.o *.mod
