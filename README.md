# Crystal Clustering
A clustering method inspired by precipitation-solubility equilibrium
Basically, it's a divisive strategy over Minimum Spanning Tree optimizing a specific criterion (Gibbs free energy in terms of physical chemistry).    
For details about the method, see my blog post [A novel clustering method - Crystal clustering](https://peace-van.github.io/climate/2023/11/01/crystalcluster.html).   

`iris.mat` is an example dataset (the Iris flower dataset). The weights `iris.W` is not used in the experiment described in the blog post.
`centroids.mat` is the dataset used in the climate clustering, described in [this post](https://peace-van.github.io/climate/2023/11/17/sec6.html).   

## Matlab
The Matlab version implements a function interface.
> [idx, mst, G, theoT] = crystalcluster(X, W, T, mode, loops, verbose)   
   
Cluster on dataset `X` with weights `W` at 'temperature' `T`, using the algorithm specified by `mode`, running a maximum number of `loops`.   
The cluster number is not directly given, but controlled by the relaxation parameter `T`. `T` needs to be positive and higher `T` gives more clusters.   
   
For definitions of the parameters, see the leading comments in `crystalcluster.m`.   

## Python
The Python version implements a class interface.

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
