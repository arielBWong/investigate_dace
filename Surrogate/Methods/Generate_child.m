% Generates offspring using DE recombination
function [X_Child] = Generate_child(pop,prob,param)
X_Child=[];tmp=pop.x;
ind1=1:size(pop.x,1);
for i=1:size(pop.x,1)
    id=randperm(size(pop.x,1)-1);
    set=setdiff(1:size(pop.x,1),ind1(i));
    ind2(i)=set(id(1));
    ind3(i)=set(id(2));
end

% Generate unique offspring: They are unique wrt to 2N
i=1;
while (size(X_Child,1)<size(pop.x,1))
    parent_order=[ind1(i) ind2(i) ind3(i)];
    child = DE(prob,pop.x(parent_order,:),param);
    X_Child=[X_Child;child];
    i=i+1;
end
return