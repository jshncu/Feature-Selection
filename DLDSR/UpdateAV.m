function [A,V,G,G1] = UpdateAV(U, X, alpha,beta,mu, DictSize,G,G1)
% Update A 
%dictsize=k;
[d,n] = size(X);
I = eye(DictSize);

AA = U'*U+mu*G;   %B in the paper
tmp = bsxfun(@times, X, G1);
if(d<n)
  C1 = (tmp*X' + (beta/alpha)*eye(d))\(tmp);
else
  C1 = tmp / (X'*tmp+(beta/alpha)*eye(n)); % L^(-1) X * (X' L^(-1) * X + \beta / \alpha I)^(-1)
end
C = X' * C1; 
C = (C + C') / 2;
tmp = eye(n) - C;
BB = alpha * (tmp * tmp');
CC = -U'*X;
A = lyap(AA,BB,CC);
V = C1 * A';

ac = 1./(2* sqrt((sum(A.*A,2)))+eps);
G = diag(ac);
G1 = 2* sqrt((sum(V.*V,2)));
end



