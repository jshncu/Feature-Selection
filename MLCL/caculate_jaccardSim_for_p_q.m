clc;clear;
dataset_name = 'MSRC_v1';

load('MSRC_v1.mat');
nSmp = size(X.data{1},1);
dist = zeros(nSmp, nSmp);
Num_view=length(X.data);
labels=cell(1,Num_view);
S=cell(1,Num_view*(Num_view-1));
%%%% Initialize K for knn
K=30;
for v = 1 : Num_view
    norm_X = sum(X.data{v}.^2, 2);
    dist =  bsxfun(@plus, bsxfun(@minus, norm_X, 2 * (X.data{v} * X.data{v}')), norm_X');
    [~,Index]=sort(dist);   
    labels{v}=Index(2:K+1,:);            
end



t=1;
S=cell(1,Num_view*(Num_view-1));
for u = 1 : Num_view
    for v = 1 : Num_view
        if u~=v
            for i=1:nSmp
                for j = [1:(i-1), (i+1):nSmp]
                    a=length(intersect(labels{u}(:,i),labels{v}(:,j)));
                    b=length(union(labels{u}(:,i),labels{v}(:,j)));
                    S{t}(i,j)=a/b;
                end
            end
            t=t+1;
        end
    end   
end



S_temp=zeros(nSmp,nSmp);
for t=1:Num_view*(Num_view-1)
    % 归一化
    S{t} = bsxfun(@rdivide, S{t}, max(S{t}, [], 2));
    S_temp=S_temp+S{t};
end
S_bar=S_temp/t;


 filename = sprintf('init_S_nor_%s.mat', dataset_name);


if exist(filename, 'file')
    load(filename);
else
    save(filename,'S','S_bar');
end


function [ProcessData]=NormalizeData1(X)

[nSmp,nFea] = size(X);
    for i = 1:nSmp
         ProcessData(i,:) = X(i,:) ./ max(1e-12,norm(X(i,:)));
    end
end