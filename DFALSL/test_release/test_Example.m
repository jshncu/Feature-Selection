clear all; 

dataset_path = sprintf('../isolet5.mat');
load(dataset_path);
X=samples;

[r,c] = size(X);

param.D=zeros(r);
param.mode=2;
param.modeD=0;
param.lambda=0.15;
param.numThreads=-1; % number of threads
param.batchsize=400;
param.verbose=false;

param.iter=1000;  % let us see what happens after 1000 iterations.

B = mexTrainDL(X,param);

