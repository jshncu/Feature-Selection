function [ W ] = updateW( X, W, S, beta, gamma,k)
view_num = length(X);
n=size(X{1},2);
Ls=(eye(n,n)-S')*(eye(n,n)-S);
Ls = beta*(Ls + Ls') / 2;
Y = eig1(Ls, k, 0);
for v=1:view_num%
    W{v}= LS21(X{v}', Y, gamma);
end
end

function [ value ] = objectiveValue2(X,W,S,beta,gamma)
view_num = length(X);
value=0;
n=size(X{1},2);
    for v=1:view_num
        temp1=W{v}'*X{v}*(eye(n,n)-S');
        temp2=W{v};
        temp_value=beta * sum(sum(temp1.^2, 2)) + ...
            gamma * sum(sum(temp2.^2, 2).^(0.5));
        value=value+temp_value;
    end
end