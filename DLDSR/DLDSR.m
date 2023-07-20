function  [V]=DLDSR(X,U,alpha,beta,mu,DictSize)

%% 
% The function is to sove the following optimization problem 
% min_{U,V,A} ||X-UA||_F^2+alpha*||A-V'X||_F^2+beta*||V||_{2,1}+mu*||A||_{2,1}
% Input:
%X : feature matrix  d*n  d is feature dimension  n is sample number 
%beta parameter : default value 1 
%alpha parameter :grid-search
%mu  parameter :grid-search
%DictSize  dictionary size 

% Output:
%V analysis dictionary    d*k

%%
X = NormalizeFea(X,1);
iternum=100;
d=size(X,1);

G = eye(DictSize);
G1 = ones(d,1);
error = zeros(iternum, 1);
for i=1:iternum
%  Alternatively update V, U and A
    [A,V,G,G1] = UpdateAV(U, X, alpha,beta,mu, DictSize,G,G1);
   	U = UpdateU(A, X, U);                 %update U
   	error(i) = objValue(X, U, A, V, alpha, beta, mu);
   	if i>1 && (error(i-1)-error(i))/error(i-1)<1e-6
    	break;
   	end
end
end

function obj = objValue(X, U, A, V, alpha, beta, mu)
E1 = X-U*A;
E2 = A-V'*X;
vc = sqrt(sum(V.*V,2));
ac = sqrt(sum(A.*A,2));
obj = sum(sum(E1.*E1))+alpha*sum(sum(E2.*E2))+beta*sum(vc)+mu*sum(ac);
end


