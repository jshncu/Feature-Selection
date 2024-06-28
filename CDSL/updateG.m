function [ G ] = updateG( Z, X, D1, G)
view_num = length(X);
rho = 1;
rate_rho = 1.2;
Iter = 1;ERROR=1;
[R, Sigma]=eig(Z*Z');
Sigma=diag(Sigma)';
TempG     = G;
TempH     = G;
TempT     = zeros(size(TempH));
previousG = G;
while(ERROR>1e-8&&Iter<1000)

    TempG   = updateTempG( Z, X, TempH, TempT, D1, R, Sigma, rho);
    TempH   = normcol_lessequal(TempG+TempT);
    TempT   = TempT+TempG-TempH;
    rho     = rate_rho*rho;
    ERROR = max(max(abs(previousG- TempG)));
    previousG = TempG;
    Iter=Iter+1;
end
G = TempG;
end
