# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 00:02:05 2023

@author: Tracy
"""


# Crystal clustering based on binary search of `T` for a desired number of clusters `target_k`
# Search range (0, 2 * model.theoT)
def binary_search_T(model, target_k, verbose=False):
    low = 0
    high = model.theoT * 2
    while True:
        curr = (low + high) / 2
        model.reset(new_temp=curr)
        idx = model.fit_predict(verbose=False)
        n_comps = idx.max() + 1
        if verbose:
            print('Temperature: %.1f, N_clusters: %d' % (curr, n_comps))
        if n_comps == target_k:
            break
        elif n_comps < target_k:
            low = curr
        else:
            high = curr
    return curr, idx


if __name__ == '__main__':
    from scipy.io import loadmat
    from CrystalCluster import CrystalCluster

    iris = loadmat('iris.mat')['iris']
    X = iris['X'][0][0]
    W = iris['W'][0][0].flatten()
    k = 3

    # Initialize with data
    cc = CrystalCluster(X, weights=None, temperature=None, k=None)
    # The theoretical temperature
    print(cc.theoT)
    T, idx = binary_search_T(cc, k, verbose=True)
