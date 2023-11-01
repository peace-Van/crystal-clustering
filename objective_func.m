function G = objective_func(m, mst, W, T)
    m = (m > 0.5);
    sub = rmedge(mst, find(m));
    G = sum(sub.Edges.Weight) - T * my_entropy(sub, W);
end