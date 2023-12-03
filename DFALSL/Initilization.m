function [U, A] = Initilization( X,mu, DictSize)
% In this intilization function, we do the following things:
% 1. Random initialization of dictioanry pair U for each class   

n = size(X,2);
param.K = DictSize;  
param.mode = 2;
param.modeD = 0;
param.lambda = mu;
param.lambda2 = 0;
param.numThreads = -1;
param.batchsize = n;
param.verbose = false;
param.iter = 100;

U = mexTrainDL_Memory(X,param);   %dictionary initialization d*k
A = pinv(U' * U) * U' * X;
end

