function [W] = CDSL(Xv,X,p,alpha,beta, gamma,G,c,r)
%% input: 
% Xv:multi-view dataset,size(X{v})= [d(v),N]  
% X: concatenated data,size(X)= [d,N]  
% s: Dimension of the collaborative subspace 
% c: Dimension of the discriminative subspaces
%% output:W
%% 1: initial parameter variables
 [d, n] = size(X);
 view_num = length(Xv);
 W=cell(1,view_num);
 Z=rand(p,n);
 dv=zeros(1,view_num);
 D1 = ones(d, 1);
 S=zeros(n,n);
 S(:,:)=1/(n-1);
 S=S-diag(diag(S));
 for v=1:view_num
   dv(v)=size(Xv{v},1);
   W{v}=rand(dv(v),c);
 end
 Omega= repmat(1/view_num,view_num,1);
 epsil = 1e-6;   % convergence threshold 
 num = 0;  %iteration number flag 
 iternum=100;
% eps = 0.0001;
 %% 2: optimization 
 for i=1:iternum
    %% Update Z
    Z=updateZ(D1, X, G, S, alpha);

    %% Update G 
    G=updateG( Z, X, D1, G);
    D1 = 1 ./ (2 * sum((X-G*Z).^2, 2).^(0.5)+eps);
 
    %% Update W
    W =updateW( Xv, W, S, beta, gamma,c);
    
    for v=1:view_num
        D2{v} = 2 * sum(W{v}.^2, 2).^(0.5); 
    end
     %% Update S
     S=updateS(Xv,W,S,Z,Omega,alpha,beta,r);%\sum(dv*k*n^2)

  %% Update Omaga
     Omega=updateOmega( Xv, W, S, r);

    %% calculate objective function value
    value(i)=objectiveValue(Xv,X,G,Z,W,S,Omega,alpha,beta,gamma,r);     
    if(i > 1)
        fprintf('val(%d):%.10f, val(%d):%.10f\t\t%.10f\n', i - 1, value(i - 1), ...
            i, value(i), value(i - 1) - value(i));
    end
    num = num + 1;
    %% 9:convergence condition
    if(i>1 && ((value(i-1)-value(i))/ value(i))<epsil)
            break;
    end  
 end
end

function [ value ] = objectiveValue(Xv,X,G,Z,W,S,Omega,alpha,beta,gamma,r)
view_num = length(Xv);
n=size(X,2);
temp1=X-G*Z;
value=sum(sum(temp1.^2,2).^(0.5));
temp2=Z-Z*S';
value=value+alpha*sum(sum(temp2.^2,2));
for v=1:view_num
        temp3=W{v}'*Xv{v}*(eye(n,n)-S');
        temp4=W{v};
        temp_value=beta.*(Omega(v)^r).*sum(sum(temp3.^2,2))+gamma*sum(sum(temp4.^2,2).^(0.5));
        value=value+temp_value;
end
end


