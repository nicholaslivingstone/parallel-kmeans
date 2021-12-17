from numpy.lib.function_base import append
from numpy.random.mtrand import noncentral_chisquare
from get_data import get_data
from mpi4py import MPI
from scipy.spatial import distance
import numpy as np
import time
import csv
from os.path import exists
from sklearn.cluster import kmeans_plusplus



def parallel_kmeans(x, k, centroids_init, comm, max_iter=300, tol=1e-4):
    centroids_old = centroids_init
    
    # Calculate distance from centroids
    dist = distance.cdist(x, centroids_old, 'euclidean')

    # Identify cluster membership of each point
    labels = np.array([np.argmin(i) for i in dist])

    # Repeat iteratively
    for iteration in range(max_iter):
        centroids_new = np.zeros(centroids_init.shape)
        # Update centroids
        for i in range(k):
            temp_points = x[labels == i]
            if(temp_points.size != 0):
                temp_cent = x[labels == i].mean(axis=0)
                centroids_new[i] = temp_cent
        
        comm.Barrier()
        # All reduce and calculate average of locally computed clusters
        centroids_buf = np.empty(centroids_new.shape)
        comm.Allreduce(centroids_new, centroids_buf)
        centroids_new = centroids_buf / comm.Get_size()
        
        diff = np.linalg.norm(centroids_new-centroids_old)
        dist = distance.cdist(x, centroids_new, 'euclidean')
        labels = np.array([np.argmin(i) for i in dist])

        # Successfully converged
        if(diff < tol):
            # print(f"Converged Successfully in {iteration} iterations. Diff = {diff}")
            return 1, 0, iteration+1

        centroids_old = centroids_new

    return 0, 0, max_iter


def init_centroids(x, k):
    """Randomly generates centroids based on a given dataset and number of clusters. 

    Args:
        x (array): Dataset
        k (int): number of clusters
        
    Returns:
        array: Generated centroids
    """    
    idx = np.random.choice(len(x), k, replace=False)
    return x[idx, :]

def all_reduce_centroids(a, b):
    return np.mean(np.array([a, b]), axis=0)
    
    

#################### SETUP MPI - START ####################
comm = MPI.COMM_WORLD		#Communication framework
root = 0			        #Root process
rank = comm.Get_rank()		#Rank of this process
num_procs = comm.Get_size()	#Total number of processes
########################### END ############################

data = None
dim = None 
n = None
k = None
centroids_init = None
sendcounts = None

if(rank == root):
    data, labels, k = get_data() # Gather data from matrix
    dim = data.shape
    n = int(np.ceil(dim[0] / num_procs))  # Number of data points each process will receive
    # centroids_init = init_centroids(data, k)
    centroids_init, indeces = kmeans_plusplus(data, k)
    print('Items per process: ', n)
    # if not n.is_integer():
    #     m = (n - np.floor(n)) * num_procs
    #     sendcounts = np.full(num_procs, n)
    #     sendcounts[:m] += 1

        
# Broadcast necessary data to each process
k = comm.bcast(k, root)
dim = comm.bcast(dim, root)
n = comm.bcast(n, root)     


comm.Barrier()
if rank != root:
    centroids_init = np.empty([k, dim[1]])
data_buffer = np.empty([n,dim[1]]) # Allocate space for data chunks

comm.Barrier()
comm.Bcast(centroids_init, root) # Send initial centroids
comm.Barrier()
comm.Scatterv(data, data_buffer, root)      # Gather data on each process

t0 = MPI.Wtime()
converged, t_total, iters = parallel_kmeans(data_buffer, k, centroids_init, comm)
t_final = MPI.Wtime()
if(rank == root):
    t_total = t_final-t0
    if converged:
        print(f"Converged successfully in {iters} iterations")
    else:
        print("Reached max iterations")
    print('Time Elapsed: %.5f seconds.' % (t_total))
    
    if not exists('parallel_kmeans_results.csv'):
        with open(r'parallel_kmeans_results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['procs', 'n', 'time', 'iters', 'converged'])

    with open(r'parallel_kmeans_results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([num_procs, data.shape[0], t_total, iters, converged])
        

