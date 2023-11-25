# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 20:54:15 2023

@author: Tracy
"""


from scipy.spatial.distance import pdist, cdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, depth_first_order
from link_cut_tree import SubtreeSumNode
import numpy as np


class CrystalCluster:
    def __init__(self, temperature, k, X, weights=None):
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
        self.graph = -minimum_spanning_tree(dists)

        # Choose the node closest to the centroid as tree root
        C = np.mean(self.data, axis=0, keepdims=True)
        dists_ = cdist(C, self.data)
        r = sorted([(b, a) for a, b in enumerate(dists_[0, :])])
        for _, i in r:
            if self.graph[i, :].nnz > 1:
                self.root = i
                break

        # Construct the link-cut tree
        _, self.graph_edges = depth_first_order(
            self.graph, self.root, directed=False)
        self.dH = np.zeros(self.N, dtype=np.float64)
        self.nodes = [SubtreeSumNode(i, self.W[i]) for i in range(self.N)]
        for u, v in enumerate(self.graph_edges):
            # v is the predecessor of u
            if v >= 0:
                if self.graph[u, v]:
                    self.dH[u] = self.graph[u, v]
                else:
                    self.dH[u] = self.graph[v, u]
                self.nodes[u].lc_link(self.nodes[v])
            else:
                self.dH[u] = np.inf

        self.theoT = np.sum(self.graph) / np.log(self.N)
        self.graph_ori = self.graph.todok().astype(bool, copy=False)

        self.reset(temperature, k)

    # Reset the system to starting state, and optionally set a new temperature
    def reset(self, new_temp=None, new_k=None):
        disconnections = np.nonzero(self.dH < 0)[0]
        for u in disconnections:
            v = self.graph_edges[u]
            self.nodes[u].lc_link(self.nodes[v])

        self.dH = np.abs(self.dH)
        self.dS = np.zeros(self.N, dtype=np.float64)
        self.graph = self.graph_ori.copy()

        for edge in range(self.N):
            self._update_dS(edge)

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
            self.n_comps = 1

    # Calculate the dS of an action
    # O(log N) thanks to link-cut tree
    def _update_dS(self, edge):
        # v is the predecessor of u
        u = edge
        if u < 0:
            return
        v = self.graph_edges[u]
        if v < 0:
            return

        if self.dH[edge] > 0:
            # if u and v is connected
            N3 = self.nodes[u].lc_get_root().get_sum()
            N1 = self.nodes[u].get_sum()
            N2 = N3 - N1
            self.dS[edge] = -N1/self.N * np.log(N1/self.N) - N2/self.N * np.log(
                N2/self.N) + N3/self.N * np.log(N3/self.N)
        else:
            # if u and v is disconnected
            N1 = self.nodes[u].get_sum()
            N2 = self.nodes[v].lc_get_root().get_sum()
            N3 = N1 + N2
            self.dS[edge] = -N3/self.N * np.log(N3/self.N) + N1/self.N * np.log(
                N1/self.N) + N2/self.N * np.log(N2/self.N)

    # One action
    def _loop(self, verbose=False):
        # Find the action with the lowest negative dG or Tm
        if self.mode == 't':
            edge = self.dG.argmin()
        elif self.mode == 'k':
            edge = (self.Tm / (self.dH > 0)).argmin()
        u = edge
        if u < 0:
            return
        v = self.graph_edges[u]
        if v < 0:
            return

        flag = self.dH[edge] > 0

        # Find the edges engaging the nodes in the connected components
        # of the two nodes (u, v)
        # Once the action is done, the dS of these edges will change
        # DFS is faster than a brute-force `connected_components` at later stage of training
        if flag:
            # if the edge is to be cut
            affected_nodes = depth_first_order(
                self.graph, u, directed=False, return_predecessors=False).tolist()
        else:
            # if the edge is to be linked
            affected_nodes_u = depth_first_order(
                self.graph, u, directed=False, return_predecessors=False).tolist()
            affected_nodes_v = depth_first_order(
                self.graph, v, directed=False, return_predecessors=False).tolist()
            affected_nodes = affected_nodes_u + affected_nodes_v
        affected_nodes_ = np.nonzero(
            np.isin(self.graph_edges, affected_nodes, assume_unique=True))[0].tolist()
        affected_edges = set(affected_nodes + affected_nodes_)

        if verbose:
            if flag:
                print('break %d - %d' % (u, v))
            else:
                print('connect %d - %d' % (u, v))

        # connect/break the edge
        if self.mode == 't':
            self.score += self.dG[edge]
        elif self.mode == 'k':
            self.n_comps += 1

        if flag:
            self.nodes[u].lc_cut()
        else:
            self.nodes[u].lc_link(self.nodes[v])
        if (u, v) in self.graph:
            self.graph[u, v] = not self.graph[u, v]
        else:
            self.graph[v, u] = not self.graph[v, u]

        # Update dH, dS, dG, Tm after the action
        self.dH[edge] = -self.dH[edge]

        for edge in affected_edges:
            self._update_dS(edge)

        if self.mode == 't':
            self.dG = self.dH - self.T * self.dS
        elif self.mode == 'k':
            self.Tm = self.dH / self.dS

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
        if self.mode == 't':
            # no entry in dG is negative
            return not np.any(self.dG < 0)
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

    # Initialize with temperature and data
    cc = CrystalCluster(5.0, None, X, weights=None)
    # Fit the model, specify max iterations (can use np.inf)
    idx = cc.fit_predict(50, verbose=True)
    # (Optional) Assign cluster index for new data
    # idx2 = cc.predict(X)

    # The theoretical temperature
    print(cc.theoT)
    # The value of the Gibbs free energy (objective function, not available for k mode)
    print(cc.score)
