# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 20:54:15 2023

@author: Tracy
"""


from scipy.spatial.distance import pdist, cdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import dok_array
import numpy as np


# Calculate the entropy of a graph
def graph_entropy(graph, W=None):
    if W is None:
        W = np.ones(graph.shape[0], dtype=np.float64)

    n, bins = connected_components(graph)

    if n == 1:
        return 0.0
    cnt = np.bincount(bins, weights=W)
    N = np.sum(W)
    return -np.sum(cnt * np.log(cnt / N)) / N


class CrystalCluster:
    def __init__(self, X, weights=None, temperature=None, k=None,
                 metric='euclidean', standardize=False, dist_lim=[0, np.inf], **kwargs):
        self.data = np.array(X)
        self.N = len(self.data)
        if weights is None:
            self.W = np.ones(self.N, dtype=np.float64)
        else:
            # weights are standardized such that sum(W) = N
            self.W = weights * self.N / np.sum(weights)

        # `dist_lim` is used to enforce certain bonds
        # Pairs with distance below or equal to `dist_lim[0]` will always be in one cluster
        # Pairs with distance above or equal to `dist_lim[1]` will never be in one cluster
        dists = squareform(pdist(self.data, metric=metric, **kwargs))
        dists[dists <= dist_lim[0]] = 0.0
        dists[dists >= dist_lim[1]] = np.inf
        w = np.expand_dims(self.W, 1)
        # The constructed MST is the same as if `dists` is directly used
        # since -1/d is a monotonically increasing function for d>0
        dists = -(w @ w.T) / dists

        # In the starting state all bonds in the MST are connected
        # so the dH are all positive
        self.dH = (-minimum_spanning_tree(dists)).todok()

        # If standardized, theoT will be 1
        if standardize:
            self.dH = self.dH / np.sum(self.dH) * np.log(self.N)

        self.theoT = np.sum(self.dH) / np.log(self.N)
        self.mst_edges = list(self.dH.keys())
        self.edges_view = np.array([list(t) for t in self.mst_edges])

        self.reset(temperature, k)

    # Reset the system to starting state, and optionally set a new temperature
    def reset(self, new_temp=None, new_k=None):
        self.dH = np.abs(self.dH).todok()
        self.graph = self.dH.astype(bool)
        self.dS = dok_array((self.N, self.N), dtype=np.float64)

        for edge in self.mst_edges:
            self._update_dS(edge)

        self.mode = None
        if new_temp is not None:
            self.T = new_temp
            self.mode = 't'
        elif new_k is not None:
            self.K = new_k
            self.mode = 'k'

        if self.mode == 't':
            self.dG = self.dH - self.T * self.dS
            self.score = 0.0
        elif self.mode == 'k':
            self.Tm = self.dH / self.dS
            self.Tm[np.isnan(self.Tm)] = 0
            self.Tm = dok_array(self.Tm)
            self.n_comps = 1

    # Calculate the dS of an action
    # brute force O(N)
    def _update_dS(self, edge):
        S0 = graph_entropy(self.graph, self.W)
        if edge not in self.mst_edges:
            edge = (edge[1], edge[0])
        self.graph[edge[0], edge[1]] = not self.graph[edge[0], edge[1]]

        S1 = graph_entropy(self.graph, self.W)

        self.dS[edge[0], edge[1]] = S1 - S0
        self.graph[edge[0], edge[1]] = not self.graph[edge[0], edge[1]]

    # One action
    def _loop(self, verbose=False):
        # Find the action with the lowest negative dG
        if self.mode == 't':
            idx = self.dG.argmin()
            row, col = np.unravel_index(idx, self.dG.shape)
        elif self.mode == 'k':
            t = self.Tm / (self.dH > 0)
            t[np.isnan(t)] = np.inf
            idx = t.argmin()
            row, col = np.unravel_index(idx, self.Tm.shape)

        if (row, col) not in self.mst_edges:
            row, col = col, row

        # Find the edges engaging the nodes in the connected components
        # of the two nodes (row, col)
        # Once the action is done, the dS of these edges will change
        bins = self.curr_state()
        affected_nodes = np.logical_or(bins == bins[row], bins == bins[col])
        affected_edges = np.nonzero(np.logical_or(
            affected_nodes[self.edges_view[:, 0]], affected_nodes[self.edges_view[:, 1]]))[0]

        # connect/break the edge
        self.graph[row, col] = not self.graph[row, col]
        if self.mode == 't':
            self.score += self.dG[row, col]
        elif self.mode == 'k':
            self.n_comps += 1

        if verbose:
            if self.dH[row, col] < 0:
                print('connect %d - %d' % (row, col))
            else:
                print('break %d - %d' % (row, col))

        # Update dH, dS, dG after the action
        self.dH[row, col] = -self.dH[row, col]

        for edge in affected_edges:
            self._update_dS(self.mst_edges[edge])

        if self.mode == 't':
            self.dG = self.dH - self.T * self.dS
        elif self.mode == 'k':
            self.Tm = self.dH / self.dS
            self.Tm[np.isnan(self.Tm)] = 0
            self.Tm = dok_array(self.Tm)

    def fit_predict(self, max_loops=np.inf, verbose=False):
        loop_cnt = 0
        while not self.is_fitted():
            self._loop(verbose)
            loop_cnt += 1
            if loop_cnt == max_loops:
                break

        if verbose:
            if self.is_fitted():
                print('Crystal clustering converged at Action %d.' % loop_cnt)
            else:
                print('Crystal clustering did not converge within %d actions.' % loop_cnt)

        return self.curr_state()

    def centroids(self):
        bins = self.curr_state()
        w = np.expand_dims(self.W, 1)
        res = []
        for i in range(np.max(bins) + 1):
            W = w * (np.expand_dims(bins == i, 1))
            res.append(self.data.T @ W / W.sum())
        return np.concatenate(res, axis=1).T

    def predict(self, X, mode='linkage'):
        if mode == 'linkage':
            bins = self.curr_state()
            dists = cdist(X, self.data)
            res = dists.argmin(axis=1)
            return bins[res]
        elif mode == 'centroid':
            centroids = self.centroids()
            dists = cdist(X, centroids)
            return dists.argmin(axis=1)

    # To save memory
    # Use of this function disables `predict` and `centroids` functionality
    # You may save the centroids beforehand
    def clear_data(self):
        del self.data

    def is_fitted(self):
        if self.mode == 't':
            # no entry in dG is negative
            return (self.dG < 0).nnz == 0
        elif self.mode == 'k':
            return self.n_comps == self.K

    def curr_state(self):
        _, bins = connected_components(self.graph)
        return bins


if __name__ == '__main__':
    from scipy.io import loadmat
    centroids = loadmat('centroids.mat')['centroids']
    X = centroids['X'][0][0]
    W = centroids['W'][0][0].flatten()

    # Initialize with `k` and data
    cc = CrystalCluster(X, weights=None, temperature=None,
                        k=26, standardize=True)
    # Fit the model, max_loops not needed for `k` mode
    idx = cc.fit_predict(verbose=True)
    # (Optional) Assign cluster index for new data
    # idx2 = cc.predict(X, mode='centroid')

    # The theoretical temperature
    print(cc.theoT)
    # The value of the Gibbs free energy (objective function, not available for `k` mode)
    # print(cc.score)
