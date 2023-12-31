# Crystal Clustering

![p](https://github.com/peace-Van/peace-Van.github.io/blob/main/assets/NN6/theo2.6.png)   

A clustering method inspired by precipitation-solubility equilibrium   

Basically, it's a divisive strategy over Minimum Spanning Tree optimizing a specific criterion (Gibbs free energy in terms of thermodynamics) and is adept at keeping the intrinsic structure of the data.   
It achieves an average normalised clustering accuracy of 0.874 (standard deviation 0.219, details in `scores_T.xlsx`) over the 73 small datasets [described by Marek Gagolewski](https://genieclust.gagolewski.com/weave/benchmarks_details.html), outperforming all the listed methods on this single metric.   
For details about the method, see my blog post [A novel clustering method - Crystal clustering](https://peace-van.github.io/climate/2023/11/01/crystalcluster.html).   

`iris.mat` is an example dataset (the Iris flower dataset). The weights `iris.W` is not used in the experiment described in the blog post.   
`centroids.mat` is the dataset used in the climate clustering. An in-depth analysis based on the dataset is provided in [this post](https://peace-van.github.io/climate/2023/11/17/sec6.html).  

## Python
The Python version uses a class interface. `CrystalCluster_brute.py` implements the brute-force algorithm, and `CrystalCluster.py` implements the link-cut tree algorithm. The interface is the same.   
The function `binary_search_T` is used to automatically search for the temperature value `T` on a dataset given the desired number of clusters. I recommend a manual check on `T`'s neighborhood after using this function. This may fine-tune the cluster morphology for the best outcome. 

> The score mentioned above is obtained by directly applying this function to each dataset and desired `k` without manual fine-tuning.   

```
# Initialize with temperature and data
cc = CrystalCluster(X, weights=None, temperature=5.0, k=None)
# Fit the model, specify max iterations (defaults `np.inf` ok in most cases)
idx = cc.fit_predict(verbose=True)
# (Optional) Assign cluster index for new data
# idx2 = cc.predict(X, mode='linkage')

# The theoretical temperature
print(cc.theoT)
# The value of the Gibbs free energy (objective function, not available for `k` mode)
print(cc.score)
```

> All distance measures supported by [`SciPy pdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) are supported. Specify using the `metric` argument and keyword arguments in the initialization of the class.   

> Not described in the blog post, an algorithm that accepts number of clusters `k` as input instead of the relaxation strength parameter `T` is also implemented. In brief, it breaks the MST bond with the lowest $\Delta H / \Delta S$ for each action until the desired number of clusters is reached. In this algorithm the `backtrack` mechanism cannot be implemented (otherwise there could be multiple clustering results for one specified $k$). Also, lowest $\Delta H / \Delta S$ is not equivalent to lowest negative $(\Delta H - T \Delta S)$ for a given positive $T$; hence the bond breaking order can be different from the algorithm using `T`, leading to different clustering results. This algorithm guarantees convergence and is `O(kN log N)` in time with link-cut tree, but fundamentally it overthrows the essence of thermodynamics equilibrium. The test results also show a slight downgrade (average NCA 0.870, std 0.227, details in `scores_k.xlsx`).   

> The induction functionality (`cc.predict`) is not mentioned in the blog post. The crystal clustering method is not designed for that, but there are simple ways to do it. Following the idea of single linkage, just assign the new data the cluster of its nearest neighbor in the training dataset, which is the `linkage` mode (`1 nearest neighbor` classification on the whole training set). Alternatively, for each cluster, compute the cluster centroid as the weighted average of its data, and assign the new data the cluster of its nearest neighbor among the cluster centroids, which is the `centroid` mode (`1 nearest neighbor` classification on the cluster centroids, which is used in [the climate classification](https://peace-van.github.io/climate/2023/11/23/sec7.html)).   

### Brute-force algorithm

The graph is stored as `SciPy`'s `Dictionary of Keys` sparse matrix. To update an entry in the `dS` matrix, the graph entropy is calculated twice. Once before and once after the action, and then subtract, which is `O(N)` time complexity. The code intends to provide an easy understanding of the mechanism.   

### Link-cut tree algorithm

The graph is stored using `SubtreeSumNode`. To update an entry in the `dS` matrix, we retrieve the sizes of the two connected components on both sides of the edge and calculate the entropy change directly. With the link-cut tree data structure, this is done in `O(log N)` time.   

> `link_cut_tree.py` is from [Asger Hautop Drewsen](https://github.com/tyilo/link_cut_tree/).

## Matlab
The Matlab version uses a function interface and implements the brute-force `greedy-backtrack` algorithm. The `surrogate`, `greedy` and `random` algorithms are for research purposes.   
> [idx, mst, G, theoT] = crystalcluster(X, W, T, mode, loops, verbose)   
   
Cluster on dataset `X` with weights `W` at 'temperature' `T`, using the algorithm specified by `mode`, running a maximum number of `loops`.   
The cluster number is not directly given, but controlled by the relaxation parameter `T`. `T` needs to be positive and higher `T` gives more clusters.   
   
For definitions of the parameters, see the leading comments in `crystalcluster.m`.   
