function [s, CostNew] = ReducedGradient(Z, WX, z, wx, s,Omega, alpha, beta,r,view_num, GradNew, CostNew)
% Z=Z(:, idx), z=Z(:, i), s=S(i, idx),WX=W'X(:, idx),wx=W'X(:,i)
%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%
goldensearch_deltmax = 1e-1;
numericalprecision = 1e-8;
d = length(s);
gold = (sqrt(5)+1)/2 ;


sInit = s ;
sNew  = sInit ; 

NormGrad = GradNew*GradNew';%按行更新
GradNew=GradNew/sqrt(NormGrad);
CostOld=CostNew;
%---------------------------------------------------------------
% Compute reduced Gradient and descent direction
%%--------------------------------------------------------------

[val,coord] = max(sNew); % coord is the index of the largest component of vector s.
GradNew = GradNew - GradNew(coord) ; % substract GradNew(coord).
desc = - GradNew.* ( (sNew>0) | (GradNew<0) ) ; % the Eq. (14) in the paper
desc(coord) = - sum(desc);  % NB:  GradNew(coord) = 0

%----------------------------------------------------
% Compute optimal stepsize
%-----------------------------------------------------
stepmin  = 0;
costmin  = CostOld ;
costmax  = 0 ;
%-----------------------------------------------------
% maximum stepsize
%-----------------------------------------------------
ind = find(desc<0);
stepmax = min(-(sNew(ind))./desc(ind));
deltmax = stepmax;
if isempty(stepmax) | stepmax==0
    s = sNew;
    return
end,
if stepmax > 0.1
     stepmax=0.1;
end;
flag=0;
%-----------------------------------------------------
%  Projected gradient
%-----------------------------------------------------
while costmax<costmin;
    flag=flag+1;
	% compute the cost with new settings
	sTmp = sNew + stepmax * desc;
    %杩欎笁琛岄渶瑕佹敼锛屽墠鍥涗釜鍙傛暟涔熻鏀?% 	Xs = X2 * sTmp;
% 	costmax = sum(Xs.^2) + gamma * sum(sY.^2) - 2 * (Xs' * x2) - 2 * gamma * (sY' * y2);
%  	Xs = X2 * sTmp;
%  	sY = sTmp' * Y;%
    Zs=Z*sTmp';%%%%%%%%%%%%%%
    costmax=alpha*sum((z-Zs).^2);%%%%%%%%%%%%
    for v=1:view_num
        costmax=costmax+beta*(Omega(v)^r)*sum((wx{v}-WX{v}*sTmp').^2);
    end
    % [costmax,Alpsupaux,w0aux,posaux] = costsvmclass(K,stepmax,desc,sNew,pos,Alpsup,C,yapp,option);
	
    if costmax<costmin
        costmin = costmax;
        sNew  = sNew + stepmax * desc;
		
        desc = desc .* ( (sNew>numericalprecision) | (desc>0) ) ;
        desc(coord) = - sum(desc([[1:coord-1] [coord+1:end]]));  
        ind = find(desc<0);
        if ~isempty(ind)
            stepmax = min(-(sNew(ind))./desc(ind));
            deltmax = stepmax;
            costmax = 0;
        else
            stepmax = 0;
            deltmax = 0;
        end;        
    end;
end

%-----------------------------------------------------
%  Linesearch
%-----------------------------------------------------

Step = [stepmin stepmax];
Cost = [costmin costmax];
[val,coord] = min(Cost);
% optimization of stepsize by golden search
while (stepmax-stepmin)>goldensearch_deltmax*(abs(deltmax))  & stepmax > eps;
    stepmedr = stepmin+(stepmax-stepmin)/gold;
    stepmedl = stepmin+(stepmedr-stepmin)/gold;    
	
	sTmp = sNew + stepmedr * desc;
% 	sX = sTmp' * X2;
% 	sY = sTmp' * Y;%
   Zs=Z*sTmp';%%%%%%%%%%%%%%
   costmedr=alpha*sum((z-Zs).^2);%%%%%%%%%%%%
    for v=1:view_num
        costmedr=costmax+beta*(Omega(v)^r)*sum((wx{v}-WX{v}*sTmp').^2);
    end
% 	costmedr = sX' * sX + gamma * sY' * sY - 2 * sX' * x2 - 2 * gamma * sY' * y2;%
	sTmp = sNew + stepmedl * desc;%
% 	sX = sTmp' * X2;%
% 	sY = sTmp' * Y;%
% 	costmedl = sX' * sX + gamma * sY' * sY - 2 * sX' * x2 - 2 * gamma * sY' * y2;%
    Zs=Z*sTmp';%%%%%%%%%%%%%%%%%%%%%%
    costmedl=alpha*sum((z-Zs).^2);%%%%%%%%%%%%
    for v=1:view_num
        costmedl=costmax+beta*(Omega(v)^r)*sum((wx{v}-WX{v}*sTmp').^2);
    end
	% [costmedr,Alpsupr,w0r,posr] = costsvmclass(K,stepmedr,desc,sNew,pos,Alpsup,C,yapp,option) ;
	% [costmedl,Alpsupl,w01,posl] = costsvmclass(K,stepmedl,desc,sNew,posr,Alpsupr,C,yapp,option) ;
	
    Step = [stepmin stepmedl stepmedr stepmax];
    Cost = [costmin costmedl costmedr costmax];
    [val,coord] = min(Cost);
    switch coord
        case 1
            stepmax = stepmedl;
            costmax = costmedl;
        case 2
            stepmax = stepmedr;
            costmax = costmedr;
        case 3
            stepmin = stepmedl;
            costmin = costmedl;
        case 4
            stepmin = stepmedr;
            costmin = costmedr;
    end;
end;


%---------------------------------
% Final Updates
%---------------------------------
CostNew = Cost(coord) ;
step = Step(coord) ;
% s update
if CostNew < CostOld ;
    sNew = sNew + step * desc;      
end;       

s = sNew ;
