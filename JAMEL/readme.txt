The initializations for W and S:
    W = ones(d, m);
    S = ones(n) / (n - 1);
    S = S - diag(diag(S));

