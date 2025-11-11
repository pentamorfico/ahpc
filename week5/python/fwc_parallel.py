#!/usr/bin/env python3
"""
Applied High Performance Computing

Shallow Waters on GPUs

Assignment: Make a GPU parallelised shallow water code using OpenACC

Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen
Date:   November 2025
License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
"""

import numpy as np
import sys
import time
import argparse
from mpi4py import MPI

# Get the MPI communicator
comm = MPI.COMM_WORLD

# Get the number of processes
mpi_size = comm.Get_size()

# Get the rank of the process
mpi_rank = comm.Get_rank()


class World:
    """Representation of a flat world"""

    def __init__(self, latitude, longitude, temperature, albedo_data):
        """Create a new flat world.

        Args:
            latitude: The size of the world in the latitude dimension.
            longitude: The size of the world in the longitude dimension.
            temperature: The initial temperature (the whole world starts with the same temperature).
            albedo_data: 2D array (latitude, longitude) of albedo values.
        """
        # The current world time of the world
        self.time = 0.0
        # The size of the world in the latitude dimension and the global size
        self.latitude = latitude
        self.global_latitude = latitude
        # The size of the world in the longitude dimension
        self.longitude = longitude
        self.global_longitude = longitude
        # The offset for this rank in the latitude dimension
        self.offset_latitude = 0
        # The offset for this rank in the longitude dimension
        self.offset_longitude = 0
        # The temperature of each coordinate of the world.
        # 2D array with shape (latitude, longitude)
        self.data = np.full((latitude, longitude), temperature, dtype=np.float64)
        # The measure of the diffuse reflection of solar radiation at each world coordinate.
        # See: <https://en.wikipedia.org/wiki/Albedo>
        # 2D array with shape (latitude, longitude)
        self.albedo_data = np.array(albedo_data, dtype=np.float64)


def checksum(world):
    """Calculate checksum of the world.
    
    TODO: make sure checksum is computed globally
    Only loop *inside* data region -- not in ghostzones!
    """
    # Sum interior cells (exclude ghost zones)
    return np.sum(world.data[1:-1, 1:-1])


def stat(world):
    """Print statistics of the world.
    
    TODO: make sure stats are computed globally
    """
    # Use only interior cells (exclude ghost zones)
    interior = world.data[1:-1, 1:-1]
    mint = np.min(interior)
    maxt = np.max(interior)
    meant = np.mean(interior)
    
    print(f"min: {mint:.6f}, max: {maxt:.6f}, avg: {meant:.6f}")


def exchange_ghost_cells(world):
    """Exchange the ghost cells i.e. copy the second data row and column to the very last 
    data row and column and vice versa.
    
    Args:
        world: The world to fix the boundaries for.
    """
    # TODO: figure out exchange of ghost cells between ranks
    
    # Exchange top and bottom rows
    world.data[0, :] = world.data[-2, :]
    world.data[-1, :] = world.data[1, :]
    
    # Exchange left and right columns
    world.data[:, 0] = world.data[:, -2]
    world.data[:, -1] = world.data[:, 1]


def radiation(world):
    """Warm the world based on the position of the sun.
    
    Args:
        world: The world to warm.
    """
    sun_angle = np.cos(world.time)
    sun_intensity = 865.0
    sun_long = (np.sin(sun_angle) * (world.global_longitude / 2.0)) + world.global_longitude / 2.0
    sun_lat = world.global_latitude / 2.0
    sun_height = 100.0 + np.cos(sun_angle) * 100.0
    sun_height_squared = sun_height * sun_height
    
    # Create coordinate grids for interior cells
    i_coords = np.arange(1, world.latitude - 1)
    j_coords = np.arange(1, world.longitude - 1)
    i_grid, j_grid = np.meshgrid(i_coords, j_coords, indexing='ij')
    
    # Calculate distances using vectorized operations
    delta_lat = sun_lat - (i_grid + world.offset_latitude)
    delta_long = sun_long - (j_grid + world.offset_longitude)
    dist = np.sqrt(delta_lat**2 + delta_long**2 + sun_height_squared)
    
    # Update temperature (only interior cells)
    world.data[1:-1, 1:-1] += (sun_intensity / dist) * (1.0 - world.albedo_data[1:-1, 1:-1])
    
    exchange_ghost_cells(world)


def energy_emmision(world):
    """ Heat radiated to space """
    world.data *= 0.99


def diffuse(world):
    """ Heat diffusion """
    tmp = world.data.copy()
    
    for k in range(10):
        # 5 point stencil using vectorized operations
        # Only update interior cells
        center = world.data[1:-1, 1:-1]
        left = world.data[1:-1, :-2]
        right = world.data[1:-1, 2:]
        up = world.data[:-2, 1:-1]
        down = world.data[2:, 1:-1]
        
        tmp[1:-1, 1:-1] = (center + left + right + up + down) / 5.0
        
        # Swap arrays
        world.data = tmp.copy()
        
        exchange_ghost_cells(world)


def integrate(world):
    """One integration step at `world_time` """
    radiation(world)
    energy_emmision(world)
    diffuse(world)


def read_world_model(filename):
    """Read a world model from a HDF5 file.
       filename: The path to the HDF5 file.
    Returns:
        A new world based on the HDF5 file.
    """
    try:
        import h5py
        with h5py.File(filename, 'r') as f:
            data = f['world'][:]
            latitude, longitude = data.shape
            world = World(latitude, longitude, 293.15, data)
            print(f"World model loaded -- latitude: {latitude}, longitude: {longitude}")
            return world
    except ImportError:
        print("Error: h5py not installed. HDF5 reading not available.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading world model: {e}")
        sys.exit(1)


def write_hdf5(world, filename, iteration):
    """Write data to a hdf5 file.
    
        world: The world to write
        filename: The output filename of the HDF5 file
        iteration: The iteration number
    """
    try:
        import h5py
        with h5py.File(filename, 'a') as f:
            group = f.create_group(f"/{iteration}")
            # Data is already in 2D format (latitude, longitude)
            group.create_dataset('world', data=world.data)
    except ImportError:
        pass  # HDF5 writing not available
    except Exception as e:
        print(f"Error writing to HDF5: {e}")


def simulate(num_of_iterations, model_filename, output_filename):
    """Simulation of a flat world climate.
    
        num_of_iterations: Number of time steps to simulate
        model_filename: The filename of the world model to use (HDF5 file)
        output_filename: The filename of the written world history (HDF5 file)
    """
    # For simplicity, read in full model
    global_world = read_world_model(model_filename)
    
    # TODO: compute offsets according to rank and domain decomposition
    # Figure out size of domain for this rank
    offset_longitude = -1  # -1 because first cell is a ghostcell
    offset_latitude = -1
    longitude = global_world.longitude + 2  # one ghost cell on each end
    latitude = global_world.latitude + 2
    
    # Copy over albedo data to local world data with ghost zones
    albedo = np.zeros((latitude, longitude), dtype=np.float64)
    albedo[1:-1, 1:-1] = global_world.albedo_data
    
    # Create local world data
    world = World(latitude, longitude, 293.15, albedo)
    world.global_latitude = global_world.global_latitude
    world.global_longitude = global_world.global_longitude
    world.offset_latitude = offset_latitude
    world.offset_longitude = offset_longitude
    
    # Set up counters and loop for num_iterations of integration steps
    delta_time = world.global_longitude / 36.0
    
    begin = time.perf_counter()
    
    for iteration in range(num_of_iterations):
        world.time = iteration / delta_time
        integrate(world)
        
        # TODO: gather the Temperature on rank zero
        # Remove ghostzones and construct global data from local data
        global_world.data = world.data[1:-1, 1:-1].copy()
        
        if output_filename:
            # Only rank zero writes water history to file
            if mpi_rank == 0:
                write_hdf5(global_world, output_filename, iteration)
                print(f"{iteration} -- ", end="", flush=True)
                stat(global_world)
    
    end = time.perf_counter()
    
    stat(world)
    print(f"checksum      : {checksum(world)}")
    print(f"elapsed time  : {end - begin:.6f} sec")


"""Main function that parses the command line and start the simulation"""

# Print MPI information
processor_name = MPI.Get_processor_name()
print(f"Flat World Climate running on {processor_name}, rank {mpi_rank} out of {mpi_size}")

parser = argparse.ArgumentParser(
    description='Flat World Climate simulation',
    usage='./fwc_parallel.py --iter <number of iterations> --model <input model> --out <name of output file>'
)
parser.add_argument('--iter', type=int, required=True, help='Number of iterations')
parser.add_argument('--model', type=str, required=True, help='Input model filename (HDF5)')
parser.add_argument('--out', type=str, default='', help='Output filename (HDF5)')

args = parser.parse_args()

if args.iter <= 0:
    print("Error: iter must be positive (e.g. --iter 1000)", file=sys.stderr)
    sys.exit(1)

if not args.model:
    print("Error: You must specify the model to simulate (e.g. --model ../models/small.hdf5)", 
            file=sys.stderr)
    sys.exit(1)

simulate(args.iter, args.model, args.out)
