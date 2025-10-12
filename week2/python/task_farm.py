"""
 Applied High Performance Computing
 
 Task Farming with MPI
 
 Assignment: Make an MPI task farm. A "task" is a randomly generated integer.
 To "execute" a task, the worker sleeps for the given number of milliseconds.
 The result of a task should be send back from the worker to the master. It
 contains the rank of the worker

 Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen
 Date:   September 2025. Based on C++ code from February 2022
 License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
"""
import numpy as np
import time

# To run an MPI program we always need to include the MPI headers
from mpi4py import MPI
# Read more here: https://mpi4py.readthedocs.io/en/stable/tutorial.html

NTASKS = 5000  # number of tasks
RANDOM_SEED = 1234

def master(nworker, comm):

    # set up a random number generator and results array
    np.random.seed(RANDOM_SEED)
    task = np.random.randint(low=0, high=31, size=NTASKS)   # set up some "tasks", notice interval is [low,high[
    result = np.zeros(NTASKS, dtype=int)

    tstart = time.perf_counter()

    # simple master-worker implementation using send/recv (python object mode)
    # protocol: master sends dict {'idx': idx, 'task': taskval} to workers
    # when no more tasks remain master sends None as termination signal
    next_task = 0
    completed = 0
    status = MPI.Status()

    # initially send one task to each worker (if available)
    for worker in range(1, nworker + 1):
        if next_task < NTASKS:
            comm.send({'idx': next_task, 'task': int(task[next_task])}, dest=worker)
            next_task += 1
        else:
            # tell idle workers there's no work
            comm.send(None, dest=worker)

    # receive results and send new tasks until all are completed
    while completed < NTASKS:
        msg = comm.recv(source=MPI.ANY_SOURCE, status=status)
        src = status.Get_source()
        # expected msg: {'idx': idx, 'rank': rank}
        if isinstance(msg, dict) and 'idx' in msg and 'rank' in msg:
            result[msg['idx']] = msg['rank']
            completed += 1

            # hand out a new task if available
            if next_task < NTASKS:
                comm.send({'idx': next_task, 'task': int(task[next_task])}, dest=src)
                next_task += 1
            else:
                # no more tasks -> send termination
                comm.send(None, dest=src)

    tend = time.perf_counter()

    # Print out a status on how many tasks were completed by each worker
    if nworker == 0 : # Avoid division by zero if no workers
        print("No workers available.")
        return
    
    workdone = np.zeros(nworker, dtype=int)
    for worker in range(1, nworker + 1):
        tasksdone = 0
        for itask in range(NTASKS):
            if result[itask] == worker:
                tasksdone += 1
                workdone[worker-1] += task[itask]
        #print(f"Master: Worker {worker} solved {tasksdone} tasks with work {workdone[worker-1]} units")

    print(f'Minimum work done by a worker:', np.min(workdone))
    print(f'Maximum work done by a worker:', np.max(workdone))
    print(f'Average work done by a worker: {np.mean(workdone):.2f}')
    print(f'Std dev of work done by a worker: {np.std(workdone):.2f}')
    print(f'Expected runtime without overheads:          {np.max(workdone)/1000.0:.3f} seconds')
    print(f'Minimum runtime with perfect load balancing: {np.sum(workdone)/(nworker*1000.0):.3f} seconds')
    print(f'Runtime for master process                   {(tend - tstart):.3f} seconds')

# call this function to complete the task. It sleeps for task milliseconds
def task_function(task):
    time.sleep(task / 1000.0)  # convert milliseconds to seconds

def worker(rank, comm):
    """
    IMPLEMENT HERE THE CODE FOR THE WORKER
    Use a call to "task_function" to complete a task
    """
    status = MPI.Status()
    while True:
        data = comm.recv(source=0, status=status)
        # termination signal
        if data is None:
            break
        # data is expected to be {'idx': idx, 'task': taskval}
        idx = data.get('idx')
        taskval = data.get('task')
        # complete the task
        task_function(taskval)
        # send back the index and worker rank
        comm.send({'idx': idx, 'rank': rank}, dest=0)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    nrank = comm.Get_size()    # get the total number of ranks
    rank = comm.Get_rank()     # get the rank of this process

    if rank == 0:              # rank 0 is the master
        master(nrank-1, comm)  # there is nrank-1 worker processes
    else:                      # ranks in [1:nrank] are workers
        worker(rank, comm)
