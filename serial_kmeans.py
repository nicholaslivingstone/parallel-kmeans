from numpy.lib.function_base import append
from get_data import get_data
from mpi4py import MPI
from scipy.spatial import distance
from sklearn.cluster import kmeans_plusplus
import numpy as np
import time
import csv



def serial_kmeans(x, k, max_iter=300, tol=1e-4):
    
    # Randomly detemine centroids
    centroids_old, idx = kmeans_plusplus(x, k) 
    # Calculate distance from centroids
    dist = distance.cdist(x, centroids_old, 'euclidean')

    # Identify cluster membership of each point
    labels = np.array([np.argmin(i) for i in dist])

    # Repeat iteratively
    for iteration in range(max_iter):
        centroids_new = []

        # Update centroids
        for i in range(k):
            temp_cent = x[labels == i].mean(axis=0)
            centroids_new.append(temp_cent)
        centroids_new = np.vstack(centroids_new)
        
        print(centroids_new)

        diff = np.linalg.norm(centroids_new-centroids_old)
        dist = distance.cdist(x, centroids_new, 'euclidean')
        labels = np.array([np.argmin(i) for i in dist])

        # Successfully converged
        if(diff < tol):
            print(
                f"Converged Successfully in {iteration} iterations. Diff = {diff}")
            return 1

        centroids_old = centroids_new

    print("Reached max iterations")
    return 0


if __name__ == "__main__":
    data, labels, k = get_data()
    n = data.shape[0]
    t0 = time.time()
    converged = serial_kmeans(data, k)
    t_final = time.time()
    
    t_total = t_final - t0
    print('Time Elapsed: %.2f seconds.' % (t_total))
    
    with open(r'serial_kmeans_results.txt', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([n, t_total, converged])
        

