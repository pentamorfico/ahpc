"""Applied High Performance Computing

Task Farming with MPI

Assignment: Make an MPI task farm for analysing HEP data. To "execute" a
task, the worker computes the accuracy of a specific set of cuts.
The resulting accuracy should be send back from the worker to the master.

Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen
Date:   February 2022
License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

This file has a small CLI addition: --data to point to the CSV file and
--repeat to artificially increase the per-task work (useful for timing).
"""
import numpy as np
import time
import argparse

# To run an MPI program we always need to include the MPI headers
from mpi4py import MPI
# Read more here: https://mpi4py.readthedocs.io/en/stable/tutorial.html

# Number of cuts to try out for each event channel.
# BEWARE! Generates n_cuts^8 permutations to analyse.
# If you run many workers, you may want to increase from 3.
n_cuts = 2
n_settings = None
NO_MORE_TASKS = None

# Default repeat factor to increase per-task work (no. of times accuracy is computed)
DEFAULT_REPEAT = 1

# Class to hold the main data set together with a bit of statistics
class Data:
    def __init__(self, filename):
        self.nevents = 0
        self.name = np.array(["averageInteractionsPerCrossing", "p_Rhad", "p_Rhad1",
                              "p_TRTTrackOccupancy", "p_topoetcone40", "p_eTileGap3Cluster",
                              "p_phiModCalo", "p_etaModCalo"])
        self.data = None                    # event data
        self.NvtxReco = None               # counters; don't use them
        self.p_nTracks = None
        self.p_truthType = None            # authorative truth about a signal

        self.signal = None                 # True if p_truthType=2

        self.means_sig = np.zeros(8)       
        self.means_bckg = np.zeros(8)
        self.flip = np.zeros(8)            # flip sign if background larger than signal for type of event

        # Read events data from csv file and calculate a bit of statistics
        # Read CSV file using numpy loadtxt, skip header row
        csv_data = np.loadtxt(filename, delimiter=',', skiprows=1)
        self.nevents = csv_data.shape[0]
        
        # Extract data columns
        # Column indices: 0=line_counter, 1=averageInteractionsPerCrossing, 2=NvtxReco, 3=p_nTracks,
        # 4-10=p_Rhad through p_etaModCalo, 11=p_truthType
        self.data = np.zeros((self.nevents, 8))
        self.data[:, 0] = csv_data[:, 1]  # averageInteractionsPerCrossing
        self.NvtxReco = csv_data[:, 2].astype(int)   # counters; don't use them
        self.p_nTracks = csv_data[:, 3].astype(int)
        # Copy 7 columns : p_Rhad, p_Rhad1, p_TRTTrackOccupancy, p_topoetcone40, p_eTileGap3Cluster, p_phiModCalo, p_etaModCalo
        self.data[:, 1:8] = csv_data[:, 4:11]  # columns 4-10 in CSV
        self.p_truthType = csv_data[:, 11].astype(int)

        # Calculate mean of signal and background for eventsmeans. Signal has p_truthType = 2
        self.signal = (self.p_truthType == 2)

        self.means_sig = np.mean(self.data[self.signal, :], axis=0)
        self.means_bckg = np.mean(self.data[~self.signal, :], axis=0)

        # check for flip and change sign of data and means if needed
        for i in range(8):
            self.flip[i] = -1 if self.means_bckg[i] < self.means_sig[i] else 1
            self.data[:, i] *= self.flip[i]
            self.means_sig[i] *= self.flip[i]
            self.means_bckg[i] *= self.flip[i]

# call this function to complete the task. It calculates the accuracy of a given set of settings
def task_function(setting, ds, repeat=1):
    # pred evaluates to true if cuts for events are satisfied for all cuts
    # repeat the computation `repeat` times to increase CPU work for timing
    acc = 0.0
    for _ in range(repeat):
        pred = np.all(ds.data < setting, axis=1)
        acc = np.sum(pred == ds.signal) / ds.nevents
    return acc

def master(nworker, ds, comm, repeat=1):
    ranges = np.zeros((n_cuts, 8))  # ranges for cuts to explore

    # loop over different event channels and set up cuts
    for i in range(8):
        for j in range(n_cuts):
            ranges[j, i] = ds.means_sig[i] + j * (ds.means_bckg[i] - ds.means_sig[i]) / n_cuts

    # generate list of all permutations of the cuts for each channel
    settings = np.zeros((n_settings, 8))
    for k in range(n_settings):
        div = 1
        for i in range(8):  # get 8-dimensional coordinate in n_cut^8 space corresponding to k and store range value
            idx = (k // div) % n_cuts
            settings[k, i] = ranges[idx, i]
            div *= n_cuts

    # results vector with the accuracy of each set of settings
    accuracy = np.zeros(n_settings)

    tstart = time.time()  # start time

    # ================================================================
    # dynamic task-farm master
    next_task = 0
    completed = 0
    status = MPI.Status()

    for worker_rank in range(1, nworker + 1):
        if next_task < n_settings:
            comm.send({'idx': next_task, 'setting': settings[next_task]}, dest=worker_rank)
            next_task += 1
        else:
            comm.send(None, dest=worker_rank)

    while completed < n_settings:
        msg = comm.recv(source=MPI.ANY_SOURCE, status=status)
        src = status.Get_source()
        if isinstance(msg, dict) and 'idx' in msg and 'accuracy' in msg:
            accuracy[msg['idx']] = msg['accuracy']
            completed += 1
            if next_task < n_settings:
                comm.send({'idx': next_task, 'setting': settings[next_task]}, dest=src)
                next_task += 1
            else:
                comm.send(None, dest=src)

    tend = time.time()  # end time
    # diagnostics
    # extract index and value for best accuracy
    best_accuracy_score = 0
    idx_best = 0
    for k in range(n_settings):
        if accuracy[k] > best_accuracy_score:
            best_accuracy_score = accuracy[k]
            idx_best = k
    
    print(f"Best accuracy obtained :{best_accuracy_score}")
    print("Final cuts :")
    for i in range(8):
        print(f"{ds.name[i]:>30s} : {settings[idx_best, i] * ds.flip[i]}")
    
    print()
    print(f"Number of settings:{n_settings:>9d}")
    elapsed = tend - tstart
    print(f"Elapsed time      :{elapsed:>9.4f} s")
    print(f"task time [mus]   :{elapsed * 1e6 / n_settings:>9.4f}")
    print(f"repeat factor     :{repeat}")

def worker(rank, ds, comm, repeat=1):
    """
    IMPLEMENT HERE THE CODE FOR THE WORKER
    Use a call to "task_function" to complete a task and return accuracy to master.
    """
    status = MPI.Status()
    while True:
        msg = comm.recv(source=0, status=status)
        if msg is None:
            break
        idx = msg.get('idx')
        setting = msg.get('setting')
        acc = task_function(setting, ds, repeat=repeat)
        comm.send({'idx': idx, 'accuracy': float(acc)}, dest=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="../mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv",
                        help='CSV data file to load')
    parser.add_argument('--n_cuts', type=int, default=None,
                        help='number of cut values per channel (overrides module default)')
    parser.add_argument('--repeat', type=int, default=DEFAULT_REPEAT,
                        help='repeat factor to increase per-task work (int)')
    args = parser.parse_args()

    # allow overriding n_cuts from the command line
    if args.n_cuts is not None:
        # assign module-level n_cuts from CLI (no 'global' needed at module scope)
        n_cuts = int(args.n_cuts)

    # set settings counts based on n_cuts
    n_settings = n_cuts ** 8
    NO_MORE_TASKS = n_settings + 1

    comm = MPI.COMM_WORLD
    nrank = comm.Get_size()  # get the total number of ranks
    rank = comm.Get_rank()   # get the rank of this process

    # All ranks need to read the data
    ds = Data(filename=args.data)

    if rank == 0:        # rank 0 is the master
        master(nrank-1, ds, comm, repeat=args.repeat)  # there is nrank-1 worker processes
    else:                # ranks in [1:nrank] are workers
        worker(rank, ds, comm, repeat=args.repeat)
