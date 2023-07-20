path = '../../../../dataset/datasets/isolet/isolet5.mat'
load(path);
[r,c] = size(samples);

param.D=zeros(r);
param.mode=2;
param.modeD=0;
param.lambda=0.15;
param.numThreads=-1; % number of threads
param.batchsize=400;
param.verbose=false;

param.iter=1000;  % let us see what happens after 1000 iterations.

B = mexTrainDL(samples,param);

