function  [W, objs]=DFALSL(X, ds, G, Z, S, params) 
%% The code implements the optimization procedure for the model proposed in Dual-level Feature Assessment for Unsupervised Multi-view Feature Selection with Latent Space Learning.
% The function is to sove the following optimization problem 
% min_{G,W,Z} ||X-GZ||_F^2+alpha*||Z-W'X||_F^2+beta\sum\sum*||W'*x_i-W'*x_j||^2*S_{ij}^r+mu*||W||_{2,1}
% s.t. s'1=1 ,\|S_i\|_0=k,s_{i,:}>=0;||u_i||<=1
% Input:
%X : feature matrix  d*n  d is feature dimension  n is sample number 
%beta parameter : default value 1 
%alpha parameter :grid-search
%DictSize  dictionary size 

% Output:
%V analysis dictionary    d*k
%%

% d = sum(ds);
numViews = length(ds);
tau = ones(numViews, 1) / numViews;
alpha = params.alpha;
beta = params.beta;
mu = params.mu;
r = params.r;
k = size(G, 2);
iternum = 100;
[dim, nSmp] = size(X);
numViews = length(ds);
D = ones(dim,1);
objs = zeros(iternum + 1, 1);
W = rand(dim, k);
W = bsxfun(@rdivide, W, sqrt(sum(W.^2, 2)));
Z = rand(k, nSmp);
dist = 0;
idx = 0;
tmpDist = cell(numViews, 1);
for v = 1 : numViews
    ind = (1:ds(v)) + idx;
    WX = X(ind, :)' * W(ind, :);
    dist = dist + tau(v)^r * EuDist2(WX,WX,0);
    idx = idx + ds(v);
end
objs(1) = sum(sum((X - G * Z).^2)) + alpha * sum(sum((Z - W' * X).^2)) + ...
        beta * sum(sum(dist .* S)) * 0.5 + mu * sum(D) / 2;
numN = 10;
W = ones(dim, k);
for i=1:iternum
    % Update for W
    oW = W;
    tS = S.^r;
    tS=(tS+tS')/2;
    Ls=diag(sum(tS))-tS;  
    if dim < nSmp
        A = zeros(dim, dim);
        idx = 0;
        for v = 1 : numViews
            ind = (1:ds(v)) + idx;
            A(ind, ind) = tau(v)^r * (X(ind, :) * Ls * X(ind, :)');
            idx = idx + ds(v);
        end
        QX = (bsxfun(@times, D, X * X' + beta / alpha * A) + mu / alpha * eye(dim)) \ bsxfun(@times, D, X);
        clear A
    else
        B = zeros(dim, nSmp);
        idx = 0;
        DX = bsxfun(@times, D / mu, X); % inv(mu * D) * X     
        for v = 1 : numViews
            ind = (1:ds(v)) + idx;
            if ds(v) > nSmp % Woodbury                
                tmp = X(ind, :)' * DX(ind, :);
                tmp = beta * tau(v)^r * Ls * (tmp + tmp') / 2 + eye(nSmp);
                B(ind, :) = DX(ind, :) / tmp;
            else
                tmpC = X(ind, :) * Ls * X(ind, :)';
                tmpC = (tmpC + tmpC') / 2;

                B(ind, :) = (beta / mu * tau(v)^r * bsxfun(@times, D(ind), tmpC) + eye(ds(v))) \ DX(ind, :);
		    clear tmpC
            end
            idx = idx + ds(v);
        end
        B = alpha * B;
        tmp = eye(nSmp) + X' * B;
        tmp = (tmp + tmp') / 2;
        
        QX = B / tmp;
        clear B
        clear DX
    end
    tmp = eye(nSmp) - X' * QX;
    tmp = (tmp + tmp') / 2;
    if sum(sum(isnan(tmp))) > 0
		W = oW;
		break;
    end
	
    Z = lyap2(G' * G, alpha * tmp, - G' * X);
    W = QX * Z';
    clear QX
    D = 2 * sqrt(sum(W.*W, 2)) + eps;
    % Update for G
    if i > 0
        objVal1 = sum(sum((X - G * Z).^2));
    end
    oG = G;
    [G] = UpdateU(Z, X, G);         %update G
    if i > 0
        objVal2 = sum(sum((X - G * Z).^2));
        if objVal1 < objVal2 
            G = oG;
        end
    end
    
    %Update S
    dist = 0;
    idx = 0;
    tmpDist = cell(numViews, 1);
    for v = 1 : numViews
        ind = (1:ds(v)) + idx;
        WX = X(ind, :)' * W(ind, :);
        tmpDist{v} = EuDist2(WX,WX,0);
        dist = dist + tau(v)^r * tmpDist{v};
        idx = idx + ds(v);
    end
    dist = dist + diag(ones(nSmp, 1) * inf);
    [val, ind] = sort(dist, 'ascend');
    val = val(1 : numN, :);
    val(val < eps) = eps;
    val = val.^(1/(1-r));
    ind = bsxfun(@plus, ind(1 : numN, :), 0 : nSmp : (nSmp * (nSmp - 1)));
    S = zeros(nSmp, nSmp);
    S(ind) = bsxfun(@rdivide, val, sum(val));
    tmpTau = tau;
    for v = 1 : numViews
        tmpTau(v) = sum(sum(tmpDist{v}(ind) .* S(ind).^r));
    end
    clear tmpDist
    S = S';

    % Update tau  
    tau = (tmpTau + eps).^(1 / (1 - r));
    tau = tau / (sum(tau));
    
    %objective function
    objs(i + 1) = sum(sum((X - G * Z).^2)) + alpha * sum(sum((Z - W' * X).^2)) + ...
        beta * sum(tmpTau .* tau.^r) * 0.5 + mu * sum(D) / 2;
    clear dist

%     clc
    if (objs(i) - objs(i + 1)) / objs(i) < 1e-4
        break;
    end
end
end


