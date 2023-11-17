# Crystal Clustering
A clustering method inspired by precipitation-solubility equilibrium

> [idx, mst, G, theoT] = crystalcluster(X, W, T, mode, loops, verbose)   
   
Cluster on dataset `X` with weights `W` at 'temperature' `T`, using the algorithm specified by `mode`, running a maximum number of `loops`.   
The cluster number is not directly given, but controlled by the relaxation parameter `T`. `T` needs to be positive and higher `T` gives more clusters.   
   
Basically, it's a divisive strategy over Minimum Spanning Tree optimizing a specific criterion (Gibbs free energy in terms of physical chemistry).    
For details about the method, see my blog post [A novel clustering method - Crystal clustering](https://peace-van.github.io/climate/2023/11/01/crystalcluster.html).   
For definitions of the parameters, see the leading comments in `crystalcluster.m`.   
`iris.mat` is an example dataset (the Iris flower dataset).
The weights `iris.W` is not used in the experiment described in the blog post.
