from get_data import get_data
from mpi4py import MPI
from scipy.spatial import distance
import numpy as np
import time


def serial_kmeans(x, k, true_labels, max_iter=300, tol=1e-4):
    t0 = time.time()
    # Randomly detemine centroids
    idx = np.random.choice(len(x), k, replace=False)
    centroids_old = x[idx, :]
    centroid_labels = true_labels[idx]
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

        diff = np.linalg.norm(centroids_new-centroids_old)
        dist = distance.cdist(x, centroids_new, 'euclidean')
        labels = np.array([np.argmin(i) for i in dist])

        # Successfully converged
        if(diff < tol):
            t_final = time.time()
            print('Time Elapsed: %.2f seconds.' % (t_final-t0))
            print(
                f"Converged Successfully in {iteration} iterations. Diff = {diff}")
            return centroids_new, np.array(list(map(lambda x: centroid_labels[x], labels)))

        centroids_old = centroids_new

    print("Reached max iterations")
    return None, None


if __name__ == "__main__":
    data, labels, k = get_data()
    centroids, calc_labels = serial_kmeans(data, k, labels)
    correct = np.sum(labels == calc_labels)
    accuracy = correct / data.shape[0] * 100
    print(f'{correct} of {data.shape[0]} correctly labeled points')
    print(f'Accuracy= %{accuracy}')

