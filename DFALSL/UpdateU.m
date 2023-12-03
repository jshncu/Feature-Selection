function [ U ] = UpdateU( A, X, U)
% Update D by Eq. (12)

%matrix A£¨k by n£©
%data matrix X(d by n)
%matrix U(d by k)
[k,n] = size(A);
Imat= eye(k);  % k by k Identity matrix
In = eye(n);
TempA = A;
TempX = X;
beta = 1;
rate_beta = 1.2;
TempH = U; % introduce H
TempS = zeros(size(TempH));  %introduce dual variable S
previousU = U;
Iter = 1;
ERROR=1;
tmp1 = TempX * TempA';
tmp2 = TempA * TempA';
while(ERROR > 1e-8 && Iter < 1000)
         TempU = (beta*(TempH-TempS)+tmp1)/(beta*Imat+tmp2);  %equal(13)update U        
         TempH = normcol_lessequal(TempU+TempS);        %equal(13a)update H
         TempS = TempS+TempU-TempH;           %equal(13)update S
         beta  = rate_beta*beta;        %¸üÐÂbetaa
         %ERROR = mean(mean((previousU- TempU).^2));
         ERROR = max(max(abs(previousU- TempU)));
         % ERROR
         previousU = TempU;
         Iter=Iter+1;
end     
% Iter
U = TempU;
end
