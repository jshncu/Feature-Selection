function [ Omega] = updateOmega( X, W, S, r)
view_num = length(X);
n= size(X{1},2);
gv_sum=0;
Omega=zeros(view_num,1);
for v=1:view_num
        temp=W{v}'*X{v}*(eye(n,n)-S');
        gv=sum(sum(temp.^2,2))^(1/(1-r));
        Omega(v)=gv;
        gv_sum=gv_sum+gv;
end
Omega=Omega/gv_sum;
end
