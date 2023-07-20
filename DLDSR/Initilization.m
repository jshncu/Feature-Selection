function [U] = Initilization( X,mu, DictSize) 
n = size(X,2);
param.K = DictSize;  
param.mode = 2;
param.modeD = 0;
param.lambda = mu;
param.numThreads = -1;
param.batchsize = n;
param.verbose = false;
param.iter = 100;

U = mexTrainDL_Memory(X,param);   %dictionary initialization d*k

end

