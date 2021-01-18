% Generates offspring using DE recombination and polynomial mutation
function Offspring = DE(Prob,Parent,param)
[N,D]     = size(Parent);
CR=param.DE_crossover_rate;
F=param.DE_mutation_factor;
if(param.Poly_mutation_prob==1)
    proM=1/D;
else
    proM=param.Poly_mutation_prob;
end
disM=param.Poly_mutation_distindex;

% Differental evolution
Parent1Dec   = Parent(1:N/3,:);
Parent2Dec   = Parent(N/3+1:N/3*2,:);
Parent3Dec   = Parent(N/3*2+1:end,:);
Offspring = Parent1Dec;
Site = rand(N/3,D) < CR;
Offspring(Site) = Offspring(Site) + F*(Parent2Dec(Site)-Parent3Dec(Site));

% Polynomial mutation
Site  = rand(N/3,D) < proM/D;
mu    = rand(N/3,D);
temp1 = Site & mu<=0.5;
Lower = repmat(Prob.bounds(:,1)',N/3,1);
Upper = repmat(Prob.bounds(:,2)',N/3,1);
Offspring(temp1) = Offspring(temp1)+(Upper(temp1)-Lower(temp1)).*((2.*mu(temp1)+(1-2.*mu(temp1)).*...
    (1-(Offspring(temp1)-Lower(temp1))./(Upper(temp1)-Lower(temp1))).^(disM+1)).^(1/(disM+1))-1);
temp2  = Site & mu>0.5;
Offspring(temp2) = Offspring(temp2)+(Upper(temp2)-Lower(temp2)).*(1-(2.*(1-mu(temp2))+2.*(mu(temp2)-0.5).*...
    (1-(Upper(temp2)-Offspring(temp2))./(Upper(temp2)-Lower(temp2))).^(disM+1)).^(1/(disM+1)));

temp3 = Offspring < Lower;
temp4 = Offspring > Upper;

Offspring(temp3) = Lower(temp3);
Offspring(temp4) = Upper(temp4);
return










