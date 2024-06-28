function[S]=updateS(X,W,S,Z,Omega,alpha,beta,r)
nSmp=size(S,1);
view_num = length(X);
I=eye(nSmp,nSmp);
tempGrad=zeros(nSmp,nSmp);
for v=1:view_num
    tempGrad1=X{v}'*W{v};
    tempGrad1=(Omega(v)^r)*tempGrad1*(tempGrad1');
    tempGrad=tempGrad+tempGrad1;
end
tempGrad=2*alpha*(Z')*Z+2*beta*tempGrad;
grad=(S-I)*tempGrad;
cost=zeros(nSmp,1);
tmp = zeros(nSmp, 1);
WX=cell(1,view_num);
tempWX=cell(1,view_num);
tempWXi=cell(1,view_num);
for v=1:view_num
    WX{v}=W{v}'*X{v};
end
for i = 1 : nSmp
    
    cost(i)=alpha*sum((Z(:,i)-Z*S(i,:)').^2);
    for v=1:view_num
        cost(i)=cost(i)+beta*(Omega(v)^r)*sum((W{v}'*X{v}(:,i)-W{v}'*X{v}*(S(i,:)')).^2);
    end
    
    idx = [1 : i - 1, i + 1 : nSmp];
    for v=1:view_num
        tempWXi{v}=WX{v}(:,i);
        tempWX{v}= WX{v}(:, idx);
    end
    tic
    for loop = 1 :100 
        [S(i, idx), tmp(i)] = ReducedGradient(Z(:, idx),tempWX, Z(:, i), tempWXi, S(i, idx), Omega, alpha, beta,r,view_num, grad(i, idx), cost(i)); 
        if cost(i) - tmp(i) < 1e-3 * abs(cost(i))
            cost(i) = tmp(i);
            break;
        else
            cost(i) = tmp(i);
        end
        grad(i, :) =(S(i,:)-I(i,:))*tempGrad;
    end
end

end