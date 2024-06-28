function [ Z ] = updateZ( D1, X, G, S, alpha)
n=size(X,2);
tmp = bsxfun(@times, G', D1');
tempZ1=tmp*G;
tempZ2=-tmp*X;
tempZ1 = (tempZ1 + tempZ1') / 2;
tempZ3=alpha*(eye(n,n)-S')*(eye(n,n)-S);
Z = lyap(tempZ1, tempZ3,tempZ2);
end

