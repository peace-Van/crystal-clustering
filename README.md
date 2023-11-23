# Crystal Clustering

![p](https://github.com/peace-Van/peace-Van.github.io/blob/main/assets/NN6/theo2.6.png)   

A clustering method inspired by precipitation-solubility equilibrium   

Basically, it's a divisive strategy over Minimum Spanning Tree optimizing a specific criterion (Gibbs free energy in terms of physical chemistry).    
For details about the method, see my blog post [A novel clustering method - Crystal clustering](https://peace-van.github.io/climate/2023/11/01/crystalcluster.html).   

`iris.mat` is an example dataset (the Iris flower dataset). The weights `iris.W` is not used in the experiment described in the blog post.   
`centroids.mat` is the dataset used in the climate clustering, described in [this post](https://peace-van.github.io/climate/2023/11/17/sec6.html).   

## Python
The Python version uses a class interface. `CrystalCluster_brute.py` implements the brute-force algorithm, and `CrystalCluster.py` implements the link-cut tree algorithm. The interface is the same.   

```
 # Initialize with temperature, data and (optional) weights
 cc = CrystalCluster(5.0, X, weights=None)
 # Fit the model, specify max iterations (can use np.inf)
 idx = cc.fit_predict(50, verbose=True)
 # (Optional) Assign cluster index for new data
 idx2 = cc.predict(X)

 # The theoretical temperature
 print(cc.theoT)
 # The value of the Gibbs free energy (objective function)
 print(cc.score)
```

> The induction `cc.predict` is not mentioned in the blog post. The crystal clustering method is not designed for that, but there's a simple way to do it. Following the idea of single linkage, just assign the new data the cluster of its nearest neighbor in the training dataset.

### Brute-force algorithm

The graph is stored as `SciPy`'s `Dictionary of Keys` sparse matrix. Each time an entry in the `dS` matrix is updated, the graph entropy is calculated twice. Once before and once after the action, and then subtract, which is `O(N)` time complexity. 

### Link-cut tree algorithm

The graph is stored using `SubtreeSumNode`. Each time an entry in the `dS` matrix is updated, we retrieve the number of nodes in the two connected components on both sides of the edge and calculate the entropy change directly. With link-cut tree data structure, this is done in `O(log N)` time.   

> `link_cut_tree.py` is from [Asger Hautop Drewsen](https://github.com/tyilo/link_cut_tree/).

## Matlab
The Matlab version uses a function interface and implements the brute-force algorithm.
> [idx, mst, G, theoT] = crystalcluster(X, W, T, mode, loops, verbose)   
   
Cluster on dataset `X` with weights `W` at 'temperature' `T`, using the algorithm specified by `mode`, running a maximum number of `loops`.   
The cluster number is not directly given, but controlled by the relaxation parameter `T`. `T` needs to be positive and higher `T` gives more clusters.   
   
For definitions of the parameters, see the leading comments in `crystalcluster.m`.   
