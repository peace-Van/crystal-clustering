function ent = my_entropy(forest, W)
    bins = conncomp(forest)';
    cnt = accumarray(bins, W);
    N = sum(W);
    ent = -sum(cnt .* log(cnt / N)) / N;
end