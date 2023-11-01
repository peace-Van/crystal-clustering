function mst = my_minspantree(X, W)
    dists = squareform(pdist(X));
    dists = -(W * W') ./ dists;
    gr = graph(dists, 'omitselfloops');
    mst = minspantree(gr);
    %mst.Nodes.Weight = W;
end