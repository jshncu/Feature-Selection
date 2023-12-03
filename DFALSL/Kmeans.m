function [vObjValues, labels, P] = Kmeans(data, K, labels, tmax)
% Do the kmeans algorithm in the given subspace
% Input:
%   data: the dataset.
%   K: the number of the clusters.
%   labels: the initial label for the dataset.
%   tmax: the maximum iterations.
% Output:
%   vObjValues: the objective value of this kmeans.
%   label: the labels for the dataset in the given subspace.
%   P: cluster centers
% maxNumCompThreads(12);
t = 1;
[d, n] = size(data);
U = full(sparse(1 : n, labels, 1, n, K, n));
nc = sum(U, 1);
P = bsxfun(@rdivide, data * U, nc);
sNorm = full(sum(sum(data .* data)));
% fprintf('d: %d, n: %d, snorm: %f\n', d, n, sNorm);
norm_P = diag(P' * P);
%norm_d = sum(data .* data);
vObjValues(t) = sNorm - nc * norm_P;
while t < tmax
    t = t + 1;
    D = bsxfun(@plus, -2 .* P' * data, norm_P);
    last = labels;
    [val, labels] = min(D);
    if (all(last == labels))
        break;
    end
    vObjValues(t) = sNorm + sum(val);
    U = full(sparse(1 : n, labels, 1, n, K, n));
    nc = sum(U, 1);
    P = bsxfun(@rdivide, data * U, nc);
    norm_P = diag(P' * P);
    % output = sprintf('%d: vObjValues(%d): %f, vObjValues(%d): %f', t, ...
    %     t - 1, vObjValues(t - 1), t, vObjValues(t));
    % disp(output);
end