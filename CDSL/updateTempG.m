function [ G ] = updateTempG( Z, X, H, T, D1, R,Sigma, rho)
Gamma=bsxfun(@times, D1, X)*Z'+rho*H-rho*T;
B=1./(bsxfun(@times, D1, Sigma) +rho);
G=(Gamma*R.*B)*R';
end
