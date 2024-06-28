function [W] = MLCL(X,r1,r2,Alpha,Beta,Gamma,class_num,S,S_bar,G)
% Input:
% X: cell, each cell element stands for a data matrix on a view.
% X{v} is n*dv ;
% r1,r2,r3: a scalar value (r1,r2,r3>1)
% 正则化参数 alpha beta 
% 平衡参数 epslion

% Output:
% W: cell, each cell element stands for a data matrix on a view. 

fprintf('-- Initializing...\n');
Num_view = length(X);
nSmp = size(X{1},1); 

%% Initialization
% 为S{u,v}到S{t}的映射做准备
Num_of_S=Num_view*(Num_view-1);


% Initialize omega,gamma,lambda,delta
omega = (1/Num_view)*ones(Num_view,1);  
gamma = (1/Num_of_S)*ones(Num_of_S,1);  
lambda = (1/Num_view)*ones(Num_view,1); 
delta = (1/Num_view)*ones(Num_view,1); 

% Initialize D
dd=cell(1,Num_view);

nFea=zeros(1,Num_view);
for v=1:Num_view
    nFea(v) = size(X{v},2);
    dd{v}=ones(nFea(v),1);  
end

% Initialize k is the num of clusters
k = class_num;
S_bar = bsxfun(@rdivide, S_bar, sum(S_bar)) * k;

% Initialize S_hat
 temp11=zeros(nSmp,nSmp);
 for v=1:Num_view
      temp11=temp11+G{v};
 end
S_hat=temp11./Num_view;
S_hat = bsxfun(@rdivide, S_hat, sum(S_hat)) * k;

% Initialize W
Q=(S_bar+S_bar')/2;
L_S_bar=diag(sum(Q))-Q;
 
Q1=(S_hat+S_hat')/2;
L_S_hat=diag(sum(Q1))-Q1; 

W=cell(1,Num_view);
for v=1:Num_view
    A=2*(Alpha*omega(v)^r1*L_S_hat + Gamma*delta(v)^r2*L_S_bar);
    Y = eig1(A, k, 0);
    W{v} = LS21(X{v}, Y, Beta);
end
%% Main Loop--Alternative Optimization
epslion1 = 1e-6;
obj3(1)=0;
Max_iter = 100;
for iter =1:Max_iter
	%%  convergence condition
    obj3(iter+1)= Compute_obj_MLCL(S,X,W,S_hat,S_bar,G,r1,r2,Alpha,Beta,Gamma,omega,delta,gamma,lambda,Num_view);
    if abs(obj3(iter+1)-obj3(iter))/abs(obj3(iter))<epslion1
        break;
    end
	%%  step1 : Update W ,D    
    
Q=(S_bar+S_bar')/2;
L_S_bar=diag(sum(Q))-Q;
 
Q1=(S_hat+S_hat')/2;
L_S_hat=diag(sum(Q1))-Q1;  
for v=1:Num_view
    A=2*(Alpha*omega(v)^r1*L_S_hat + Gamma*delta(v)^r2*L_S_bar);
    Y = eig1(A, k, 0);
    W{v} = LS21(X{v}, Y, Beta,W{v});
end

	%% step 2 :update S_bar   

dist = zeros(nSmp, nSmp);
WX = cell(Num_view,1);
norm_XW=cell(Num_view,1);
for v = 1 : Num_view
    WX{v} = X{v} * W{v}; 
    norm_XW{v} = sum(WX{v}.^2, 2);
    dist = dist + delta(v)^r2 * bsxfun(@plus, bsxfun(@minus, norm_XW{v}', 2 * (WX{v} * WX{v}')), norm_XW{v});
end
for j = 1:nSmp
    A_j=dist(:,j);
    
 	candIdx = ones(nSmp, 1);
	candIdx(j) = 0;
	candIdx = candIdx > 0;
    
	formulation_part=zeros(nSmp,1);
    for t = 1:Num_of_S
        formulation_part=gamma(t)*S{t}(:,j) + formulation_part;
    end
    
    tmpVector = formulation_part(candIdx)-Alpha*0.5*A_j(candIdx);
    S_bar_temp = EProjSimplex_new(tmpVector,class_num);   
    
	obj_S_bar2=norm(S_bar_temp-tmpVector)^2;
	obj_S_bar1=norm(S_bar(candIdx,j)-tmpVector)^2;
    
    if obj_S_bar2 < obj_S_bar1
        S_bar(candIdx,j)=S_bar_temp;
    end 
end 

 %% step 3 :update S_hat   
 
dist1 = zeros(nSmp, nSmp);
for v = 1 : Num_view
    dist1 = dist1 + omega(v)^r1 * bsxfun(@plus, bsxfun(@minus, norm_XW{v}', 2 * (WX{v} * WX{v}')), norm_XW{v});
end
for j = 1:nSmp
    A_j=dist1(:,j);
    
	candIdx = ones(nSmp, 1);
	candIdx(j) = 0;
	candIdx = candIdx > 0;
    
	formulation_part=zeros(nSmp,1);
    for v = 1:Num_view
        formulation_part=lambda(v)*G{v}(:,j) + formulation_part;
    end
    
    tmpVector = formulation_part(candIdx)-Alpha*0.5*A_j(candIdx);
    S_hat_temp = EProjSimplex_new(tmpVector,class_num);   
    
	obj_S_hat2=norm(S_hat_temp-tmpVector)^2;
	obj_S_hat1=norm(S_hat(candIdx,j)-tmpVector)^2;
    
    if obj_S_hat2 < obj_S_hat1
        S_hat(candIdx,j)=S_hat_temp;
    end 
end 

	%% Step 4 : Update gamma

f=zeros(Num_of_S,1);
M=zeros(Num_of_S,Num_of_S);
for i=1:Num_of_S
    f(i)=-2*sum(sum(S_bar.*S{i}));
    for j=1:Num_of_S
        M(i,j)=sum(sum(S{i}.*S{j}));
    end
end

f=f./2;
Aeq=ones(1,Num_of_S);
beq=1;

lb=zeros(Num_of_S,1);
ub=ones(Num_of_S,1);

gamma = quadprog(M,f,[],[],Aeq,beq,lb,ub,gamma);

	%% Step 5 : Update lambda

f1=zeros(Num_view,1);
M1=zeros(Num_view,Num_view);
for i=1:Num_view
    f1(i)=-2*sum(sum(S_hat.*G{i}));
    for j=1:Num_view
        M1(i,j)=sum(sum(G{i}.*G{j}));
    end
end

f1=f1./2;
 Aeq1=ones(1,Num_view);
 beq1=1;
 
 lb1=zeros(Num_view,1);
 ub1=ones(Num_view,1);

lambda = quadprog(M1,f1,[],[],Aeq1,beq1,lb1,ub1,lambda);

	%% Step 6: Update delta
    
g_s_bar = zeros(Num_view,1);
Q=(S_bar+S_bar')/2;
L_S_bar=diag(sum(Q))-Q;   

for v=1:Num_view
    g_s_bar(v) = trace(WX{v}'*L_S_bar*WX{v});
end
t_s_bar = g_s_bar.^(1/(1-r2));

delta = zeros(Num_view,1);
for v=1:Num_view
    delta(v) = t_s_bar(v)/sum(t_s_bar);
end
    
	%% Step 7: Update omega

gg = zeros(Num_view,1); 
Q1=(S_hat+S_hat')/2;
L_S_hat=diag(sum(Q1))-Q1;  

for v=1:Num_view
	gg(v) = trace(WX{v}'*L_S_hat*WX{v});
end 
t_g = gg.^(1/(1-r2));
omega = zeros(Num_view,1);
for v=1:Num_view
    omega(v) = t_g(v)/sum(t_g);
end
    
    %% print Iteration iters
    fprintf('Iteration %d\n',iter);   
   
end  
fprintf('Optimization finished.\n');
end

function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)

if nargin < 2
    c = size(A,1);
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end;

if nargin < 3
    isMax = 1;
    isSym = 1;
end;

if nargin < 4
    isSym = 1;
end;

if isSym == 1
    A = max(A,A');
end;
try
    [v, d] = eig(A);
    d = diag(d);
    %d = real(d);
catch
    if isMax == 0 
        [v, d] = eigs(sparse(A), c, 'sa', struct('tol', 1e-5'));
    else
        [v, d] = eigs(sparse(A), c, 'la', struct('tol', 1e-5'));
    end
end

if isMax == 0
    [d1, idx] = sort(d);
else
    [d1, idx] = sort(d,'descend');
end;
idx1 = idx(1:c);
eigval = d(idx1);
eigvec = v(:,idx1);

eigval_full = d(idx);
end