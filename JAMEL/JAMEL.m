function [Y, W, obj, S]=JAMEL(X, S, W, maxIter, alpha, beta, gamma)
%	X: Rows of vectors of data points
%	S: The reconstruction matrix.
%   F: the cluster result
%   W: the feature selection matrix

n = size(X, 2);
if isempty(S)
    S = ones(n, n) / (n - 1);
    S = S - diag(diag(S));
end

[d, c] = size(W);
[nFea, nSmp] = size(X);
t = 2;

% Initialization for U
u = 2 * sqrt(sum(W .* W, 2));
L = (eye(nSmp) - S) * (eye(nSmp) - S)';
XXT = X * X';
obj = realmax;
while t <= maxIter + 1                       
    % Update for Y
    % U = spdiags(u, 0, nFea, nFea);
    if nFea <= nSmp
        UX = bsxfun(@times, u, X);
        Q = UX * L * X' + gamma * beta * (bsxfun(@times, u, XXT) + alpha * eye(nFea));
        A = Q \ (UX);
    else
        UX = 1/(alpha * beta * gamma) * bsxfun(@times, u, X); % d-by-n O(nd^2)
        A = UX / (eye(nSmp) + (L + beta * gamma * eye(nSmp)) * (X' * UX));
    end
    tmp = L - gamma * beta^2 * X' * A + beta * eye(nSmp);
    tmp = (tmp + tmp') / 2;
    [Y, D] = eig(tmp);
    clear tmp
    d = diag(D);
    [d, ind] = sort(d, 'ascend');
    Y = Y(:, ind(1 : c));
    
    % Update for W
    W = gamma * beta * A * Y;
    
    WTX = W' * X; % c-by-n

    XTWWTX = WTX' * WTX; % n-by-n O(n^2c)
    YYT = Y * Y'; % n-by-n O(n^2 c)
    WXS = WTX * S; % c-by-n O(n^2 c)
    YS = Y' * S; % c-by-n O(n^2 c)
    grad = 2 * (WTX' * WXS - XTWWTX + gamma * Y * YS - gamma * YYT); % O(c n^2)
    cost = sum(WXS.^2) + gamma * sum(YS.^2) - 2 * sum(S .* XTWWTX) - 2 * gamma * sum(S .* YYT);
    tmp = zeros(nSmp, 1);
    for i = 1 : nSmp % O(n * nc)=>O(n^2c)
        idx = [1 : i - 1, i + 1 : nSmp];
        for loop = 1 : floor((n - 1) / 101)
            
            [S(idx, i), tmp(i)] = ReducedGradient(WTX(:, idx), Y(idx, :), ...
                WTX(:, i), Y(i, :), S(idx, i), gamma, grad(idx, i), cost(i));				
            if cost(i) - tmp(i) < 1e-3 * abs(cost(i))
                cost(i) = tmp(i);
                break;
            else
                cost(i) = tmp(i);
            end
            WXS(:, i) = WTX * S(:, i);
            YS(:, i) = Y' * S(:, i);
            grad(:, i) = 2 * (WTX' * WXS(:, i) - XTWWTX(:, i) + gamma * Y * YS(:, i) - gamma * YYT(:, i));
        end            
    end
    L = (eye(nSmp) - S) * (eye(nSmp) - S)';
    
    % Update for U
    u = 2 * sqrt(sum(W .* W, 2));
    obj(t) = sum(cost) + trace(XTWWTX) + ...
        gamma * (trace(YYT) + beta * (sum(sum((WTX - Y').^2)) + ...
        alpha * sum(u) / 2));
   
    if obj(t - 1) - obj(t) < 1e-3 * obj(t - 1)
        break;
    end
    t = t + 1;
end