# parallel-kmeans
An implementation and analysis of different K-Means methods on distributed systems in python. 

[Data Source Used](https://github.com/gagolews/clustering_benchmarks_v1)

## Requirements 
* Conda 4.8.4
  * Dependencies can be found in `environment.yml` 

## How to use

### Python Script
The output of either will append the results to `parallel_kmeans_results.csv` and `serial_kmeans_results.csv`. 

#### Parallel Implementation

`mpirun -n [NUM OF PROCESSES] python parallel_kmeans.py -dn [DATA NAME] -p [PATH TO DATA]`

e.g.
`mpirun -n 4 python parallel_kmeans.py -dn dense -p ./data/graves`

Example Output
```
Attempting to gather 'dense' from './data/graves'

Loaded './data/graves/dense' n=200 d=2 k=2

Items per process:  13
Converged successfully in 3 iterations
Time Elapsed: 0.00489 seconds.
Attempting to gather 'dense' from './data/graves'

Loaded './data/graves/dense' n=200 d=2 k=2
```

#### Serial Implementation

`python serial_kmeans.py -dn [DATA NAME] -p [DATA PATH]`

### Submit as pbs job
#### Parallel

`qsub parallel_kmeans.pbs -F "-i [ITERATIONS TO RUN] -p [DATA PATH] -m [DATA NAME]" [QSUB OPTS HERE]`

e.g. `qsub parallel_kmeans.pbs -F "-i 100 -p ./data/fcps -m atom" -N pk_8_td -l nodes=1:ppn=8,walltime=00:20:00`

#### Serial
Will run by default with `nodes=1:ppn=4`

`qsub serial_kmeans.pbs -F "-i [ITERATIONS TO RUN] -p [DATA PATH] -m [DATA NAME]`
