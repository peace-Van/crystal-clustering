# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 00:02:05 2023

@author: Tracy
"""

import math


# Crystal clustering based on binary search of `T` for a desired number of clusters `target_k`
# bounds is the searching range for `T`
def binary_search_T(model, target_k, bounds=[0, 1], logarithmic=False, verbose=False):
    low = math.log10(bounds[0]) if logarithmic else bounds[0]
    high = math.log10(bounds[1]) if logarithmic else bounds[1]

    while True:
        curr = (low + high) / 2
        new_temp = 10 ** curr if logarithmic else curr
        model.reset(new_temp=new_temp)
        idx = model.fit_predict(verbose=False)
        n_comps = idx.max() + 1
        if verbose:
            print('Temperature: %.4f, Num_clusters: %d' % (new_temp, n_comps))
        if n_comps == target_k:
            break
        elif n_comps < target_k:
            low = curr
        else:
            high = curr

    return new_temp, idx


if __name__ == '__main__':
    from scipy.io import loadmat
    from CrystalCluster import CrystalCluster

    iris = loadmat('iris.mat')['iris']
    X = iris['X'][0][0]
    W = iris['W'][0][0].flatten()
    k = 3

    # Initialize with data
    cc = CrystalCluster(X, standardize=True, weights=None,
                        temperature=None, k=None)
    # The theoretical temperature
    print(cc.theoT)

    print('logarithmic binary search')
    T, idx = binary_search_T(
        cc, k, bounds=[1e-4, 1], logarithmic=True, verbose=True)
    print('linear binary search')
    T, idx = binary_search_T(
        cc, k, bounds=[0, 1], logarithmic=False, verbose=True)
