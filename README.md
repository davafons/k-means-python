# KMmeans implementation from scratch

Implementation of the __k-means__ algorithm for __clustering__ from scratch, using __Python 3.6__ and data science libraries (`numpy`, `matplotlib`, `sklearn`...)

The program need a dataset to be executed. Can be a `.txt` file, a `.csv` file, or some of the predefined datasets in `sklearn` (blobs | iris)

## Example

By default, both __this k-means implementation__ and the __sklearn k-means implementation__ are executed, showing the result of each one for comparisons. Results are provided both in __text__, and __graphically__ using `matplotlib`.

<img src="https://i.imgur.com/ahGg7PH.png" width="420" height="315" /> <img src="https://i.imgur.com/zIw2bxj.png" width="420" height="315" />

```
Personal KMeans implementation:
	Centroids:
[[6.85       3.07368421 5.74210526 2.07105263]
 [5.9016129  2.7483871  4.39354839 1.43387097]
 [5.006      3.428      1.462      0.246     ]]
	Labels:
[2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 1 0 0 0 0
 0 0 1 1 0 0 0 0 1 0 1 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 1 0
 0 1]
	Nº iter: 4
	Execution time: 0.055638000000000076


SKLearn KMeans implementation:
	Centroids:
[[5.9016129  2.7483871  4.39354839 1.43387097]
 [5.006      3.428      1.462      0.246     ]
 [6.85       3.07368421 5.74210526 2.07105263]]
	Labels:
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 2 2 2 2 0 2 2 2 2
 2 2 0 0 2 2 2 2 0 2 0 2 0 2 2 0 0 2 2 2 2 2 0 2 2 2 2 0 2 2 2 0 2 2 2 0 2
 2 0]
	Nº iter: 6
	Execution time: 0.08564799999999995
```

## Verbose output

### Run

When the `--run-verbose` parameter is passed, the result is the output of a __step by step execution of the KMeans__, until the convergence of the clusters.

![run verbose gif](https://i.imgur.com/CZjx1H4.gif)
```
Centroids                  Labels                             Iteration       SSE       CPU
-------------------------  -------------------------------  -----------  --------  --------
[[0.6 4.2]                 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]            0  620.12    0.000734
 [9.3 5.7]
 [4.5 4. ]]
[[1.         2.7       ]   [2 2 1 0 0 0 1 0 2 0 2 1 1 0 2]            1  108.828   0.000457
 [8.15       5.86666667]
 [5.82       3.04      ]]
[[1.36 3.48]               [2 2 1 0 0 0 1 0 2 0 2 1 1 0 2]            2   75.124   0.00045
 [7.94 6.8 ]
 [7.1  1.96]]
[[1.88333333 3.56666667]   [2 2 1 0 0 0 1 0 2 0 2 1 1 0 2]            3   67.3217  0.000447
 [8.05       7.45      ]
 [7.7        2.        ]]
[[1.88333333 3.56666667]   [2 2 1 0 0 0 1 0 2 0 2 1 1 0 2]            4   67.3217  0.000447
```

### Fit

When the `--fit-verbose` parameter is passed, the result will be the output of __all the KMeans instances executed in parallel__.

![fit verbose](https://i.imgur.com/RtwfTry.gif)

```
Centroids                  Labels                             Iteration       SSE       CPU
-------------------------  -------------------------------  -----------  --------  --------
[[8.1        0.73333333]   [0 0 2 1 1 1 2 1 2 1 0 2 2 1 2]            3   72.095   0.018765
 [1.88333333 3.56666667]
 [7.73333333 6.26666667]]
[[1.35       4.15      ]   [1 1 1 0 0 2 1 2 1 0 1 1 1 0 1]            2  126.103   0.000233
 [7.85555556 4.42222222]
 [2.95       2.4       ]]
[[1.36       3.48      ]   [2 2 1 0 0 0 1 2 2 0 2 1 1 0 2]            4   70.7467  0.000332
 [8.05       7.45      ]
 [7.16666667 2.33333333]]
[[8.1        0.73333333]   [0 0 2 2 1 1 2 2 2 1 0 2 2 1 2]            2   80.1304  7e-06
 [1.         2.7       ]
 [6.7125     6.025     ]]
[[1.         2.7       ]   [2 2 2 1 0 0 1 1 2 0 2 2 1 0 2]            4   89.5036  2.1e-05
 [5.125      6.825     ]
 [8.21428571 3.3       ]]
[[1.         2.7       ]   [1 1 1 2 0 0 2 2 1 0 1 1 2 0 1]            4   89.5036  0.000128
 [8.21428571 3.3       ]
 [5.125      6.825     ]]
 ```

## Options

```
usage: __main__.py [-h] [--init INIT] [--n-clusters N_CLUSTERS]
                   [--n-init N_INIT] [--n-jobs N_JOBS] [--run-verbose]
                   [--fit-verbose]
                   dataset

KMeans implementation, compared with SKLearn implementation

positional arguments:
  dataset               Path to the .txt file with the dataset to load, or one
                        of the predefined datasets: (iris, blobs).

optional arguments:
  -h, --help            show this help message and exit
  --init INIT           (random|kpp) Method used for determine the first
                        clusters
  --n-clusters N_CLUSTERS
                        Number of clusters to find
  --n-init N_INIT       Number of times to execute KMeans before returning a
                        cluster
  --n-jobs N_JOBS       Number of process to execute in parallel with KMeans
                        instances
  --run-verbose         Verbose output of only one KMeans run with the loaded
                        dataset.
  --fit-verbose         Verbose output of all the KMeans runs generated while
                        seraching thelowest SSE
```

## Author

__David Afonso Dorta__ - University of La Laguna
