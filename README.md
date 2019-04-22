# KMmeans implementation from scratch

Implementation of the __k-means__ algorithm for __clustering__ from scratch, using __Python 3.6__ and data science libraries (`numpy`, `matplotlib`, `sklearn`...)

The program need a dataset to be executed. Can be a `.txt` file, a `.csv` file, or some of the predefined datasets in `sklearn` (blobs | iris)

## Results of the KMeans execution for the example instances

This table shows the result of executing the __KMeans implementation__ for the datasets `res/prob1.txt` and `res/prob2.txt`.
(The column ___Iterations___ marks the number of iterations performed by the algorithm before convergence)

The first table is for `res/prob1.txt` and the second one for `res/prob2.txt`

| K | Iterations |   SSE   | CPU (ms) |   | K | Iterations |   SSE   | CPU (ms) |
|:-:|:----------:|:-------:|:--------:|:-:|:-:|:----------:|:-------:|:--------:|
| 2 |      2     | 244.593 | 0.018101 |   | 2 |      2     | 419.484 |  0.01929 |
| 2 |      3     | 274.877 | 0.000478 |   | 2 |      4     | 432.108 | 0.000235 |
| 2 |      3     | 260.826 |  1.1e-05 |   | 2 |      2     |  418.17 | 0.000235 |
| 2 |      2     |  247.17 | 0.000256 |   | 2 |      5     | 446.111 | 0.000149 |
| 2 |      3     | 274.877 |  2.5e-05 |   | 2 |      5     | 432.108 | 0.000166 |
| 3 |      2     | 172.455 | 0.016216 |   | 3 |      2     | 331.633 | 0.016539 |
| 3 |      2     | 194.698 |  1.2e-05 |   | 3 |      3     | 324.892 | 0.000298 |
| 3 |      3     | 180.055 |  1.4e-05 |   | 3 |      2     | 325.172 |   1e-05  |
| 3 |      3     | 174.349 |  0.00016 |   | 3 |      3     | 304.284 | 0.000177 |
| 3 |      5     | 169.868 | 0.000194 |   | 3 |      4     | 314.489 | 0.000138 |
| 4 |      2     | 96.6205 | 0.017427 |   | 4 |      2     | 281.365 | 0.017718 |
| 4 |      2     | 96.6205 | 0.000271 |   | 4 |      2     | 300.396 |  1.8e-05 |
| 4 |      2     | 96.6205 | 0.000422 |   | 4 |      2     |  319.85 |  2.5e-05 |
| 4 |      2     | 143.835 |   7e-06  |   | 4 |      3     | 261.738 | 0.000237 |
| 4 |      4     | 156.553 |   7e-06  |   | 4 |      4     | 291.523 | 0.000147 |
| 5 |      2     | 148.996 | 0.016686 |   | 5 |      3     | 226.591 | 0.017056 |
| 5 |      2     | 81.7125 | 0.000322 |   | 5 |      2     | 215.972 | 0.000242 |
| 5 |      2     | 142.791 |   8e-06  |   | 5 |      2     | 195.157 |   1e-05  |
| 5 |      3     | 81.2497 |  0.00027 |   | 5 |      3     | 246.352 |  0.00035 |
| 5 |      3     |  86.648 |   7e-06  |   | 5 |      3     | 205.223 |   8e-06  |
| 6 |      2     | 61.8283 |  0.01873 |   | 6 |      2     | 219.843 | 0.016992 |
| 6 |      2     | 67.5283 |  1.9e-05 |   | 6 |      3     | 179.916 | 0.000142 |
| 6 |      3     | 86.1903 | 0.000146 |   | 6 |      3     | 182.548 | 0.000184 |
| 6 |      3     |  60.61  | 0.000256 |   | 6 |      3     | 232.582 | 0.000247 |
| 6 |      4     |  71.665 |   7e-06  |   | 6 |      4     | 166.762 | 0.000325 |
| 7 |      2     | 60.8467 | 0.017985 |   | 7 |      3     | 154.487 | 0.017263 |
| 7 |      2     |  57.978 |   9e-06  |   | 7 |      2     |  184.35 | 0.000356 |
| 7 |      2     |  60.37  | 0.000263 |   | 7 |      2     | 173.679 |   8e-06  |
| 7 |      3     | 109.766 | 0.000306 |   | 7 |      3     | 158.412 | 0.000423 |
| 7 |      3     | 55.0292 | 0.000335 |   | 7 |      2     | 133.167 |   9e-06  |


## Examples

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
