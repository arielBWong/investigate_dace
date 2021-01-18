function [X_Child] = generate_child(lb, ub,  pop, param)
%
%
%----------------------------------
X_Child=[];
tmp=pop.X(1:param.popsize,:);
% G_Child=NaN*ones(param.popsize,prob.ng);
% G=[];
% F_Child=NaN*ones(param.popsize,prob.nf);


% This will autimatically generate ind1, ind2 and ind3 unique and ind1
% follows the order 1:N
ind1=1:param.popsize;
for i=1:param.popsize
    id=randperm(param.popsize-1);
    set=setdiff(1:param.popsize,ind1(i));
    ind2(i)=set(id(1));
    ind3(i)=set(id(2));
end


% Generate unique offspring: They are unique wrt to 2N
i=1;
while (size(X_Child,1)<param.popsize)
    child = DEoperator(lb, ub, pop.X([ind1(i);ind2(i);ind3(i)],:));
    tmp1=[tmp;child];
    [~,uni_id] = unique(tmp1,'rows','stable');
    if(length(uni_id)==size(tmp,1)+1)
        X_Child = [X_Child; child];
        tmp=[tmp;child];
        i=i+1;
    end
end
return