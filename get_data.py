# Gets data from given path and name

import sys
import os
import argparse
import numpy as np

def get_data():
    # Use this to source the script to load data
    sys.path.insert(0, '/users/neliving/cs491/parallel-kmeans/data')
    from load_dataset import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('-dn', '--data_name')
    parser.add_argument('-p', '--path')
    args = parser.parse_args()
    dataset = args.data_name
    path = args.path
    dataset = path + '/' + dataset     

    # Attempt to gather data
    print(f"Attempting to gather '{args.data_name}' from '{args.path}'\n")
    #data= load_dataset(args.data_name, args.path)
    data    = np.loadtxt(dataset+".data.gz", ndmin=2)
    labels  = np.loadtxt(dataset+".labels0.gz", dtype=np.intc)
    n = data.shape[0]
    d = data.shape[1]
    k = np.unique(labels).shape[0]
    print(f"Loaded '{dataset}' n={n} d={d} k={k}\n")
    #print(data)
    #print(labels)
    return data, labels, k  

if __name__ == "__main__":
    get_data()

