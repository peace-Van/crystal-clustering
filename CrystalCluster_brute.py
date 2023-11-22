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
    def __init__(self, temperature, X, weights=None):
        self.data = X
        self.N = len(X)
        if weights is None:
            self.W = np.ones(self.N, dtype=np.float64)
        else:
            self.W = weights * self.N / np.sum(weights)

        dists = squareform(pdist(self.data))
        w = np.expand_dims(self.W, 1)
        dists = -(w @ w.T) / dists

        # In the starting state all bonds in the MST are connected
        # so the dH are all positive
        self.dH = (-minimum_spanning_tree(dists)).todok()
        self.theoT = np.sum(self.dH) / np.log(self.N)
        self.mst_edges = list(self.dH.keys())
        self.edges_view = np.array([list(t) for t in self.mst_edges])

        self.reset(temperature)

    # Reset the system to starting state, and optionally set a new temperature
    def reset(self, new_temp=None):
        self.dH = np.abs(self.dH).todok()
        self.graph = self.dH.astype(bool)
        self.dS = dok_array((self.N, self.N), dtype=np.float64)

        for edge in self.mst_edges:
            self._update_dS(edge)

        if new_temp is not None:
            self.T = new_temp

        self.dG = self.dH - self.T * self.dS
        self.score = 0.0

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
        idx = self.dG.argmin()
        row, col = np.unravel_index(idx, self.dG.shape)
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
        self.score += self.dG[row, col]

        if verbose:
            if self.dH[row, col] < 0:
                print('connect %d - %d' % (row, col))
            else:
                print('break %d - %d' % (row, col))

        # Update dH, dS, dG after the action
        self.dH[row, col] = -self.dH[row, col]

        for edge in affected_edges:
            self._update_dS(self.mst_edges[edge])

        self.dG = self.dH - self.T * self.dS

    def fit_predict(self, max_loops, verbose=False):
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

    def predict(self, X):
        bins = self.curr_state()
        dists = cdist(X, self.data)
        res = dists.argmin(axis=1)
        return bins[res]

    def is_fitted(self):
        # no entry in dG is negative
        return (self.dG < 0).nnz == 0

    def curr_state(self):
        _, bins = connected_components(self.graph)
        return bins


if __name__ == '__main__':
    from scipy.io import loadmat
    X = loadmat('centroids.mat')['c']

    # Initialize with temperature and data
    cc = CrystalCluster(5.0, X, weights=None)
    # Fit the model, specify max iterations (can use np.inf)
    idx = cc.fit_predict(50, verbose=True)
    # (Optional) Assign cluster index for new data
    idx2 = cc.predict(X)

    # The theoretical temperature
    print(cc.theoT)
    # The value of the Gibbs free energy (objective function)
    print(cc.score)
