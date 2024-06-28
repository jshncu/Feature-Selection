function obj3= Compute_obj_fangfa111(S,X,W,S_hat,S_bar,G,r1,r2,Alpha,Beta,Gamma,omega,delta,gamma,lambda,Num_view)
     Num_of_S=Num_view*(Num_view-1);
     gvw_G  = zeros(Num_view,1);
     gvw_S_bar  = zeros(Num_view,1);

     sumDD  = zeros(Num_view,1);
     WX=cell(1,Num_view);
     nSmp=size(X{1},1);
     
      Q=(S_bar+S_bar')/2;
      L_S_bar=diag(sum(Q))-Q;
      
     Q1=(S_hat+S_hat')/2;
     L_S_hat=diag(sum(Q1))-Q1; 

    
    for v=1:Num_view 
        WX{v}=X{v}*W{v};
        gvw_G(v) = 2*trace(WX{v}'*L_S_hat*WX{v});
        gvw_S_bar(v) = 2*trace(WX{v}'*L_S_bar*WX{v});
        sumDD(v) = sum(sqrt(sum(W{v}.^2,2)));
    end
    
    S_temp=zeros(nSmp,nSmp);
    S_temp_1=zeros(nSmp,nSmp);
    for t=1:Num_of_S
        S_temp=S_temp+gamma(t)*S{t};
        S_temp_1=S_temp_1+lambda(v)*G{v};
    end
    
   % Compute the objective values
    obj3 =Alpha*(gvw_G'*omega.^r1+Gamma*gvw_S_bar'*delta.^r2) + norm(S_hat-S_temp_1,'fro')^2 + Gamma*norm(S_bar-S_temp,'fro')^2 + Beta*sum(sumDD); 
end