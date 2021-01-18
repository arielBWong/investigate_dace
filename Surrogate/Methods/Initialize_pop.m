% Initializing a population
function pop=Initialize_pop(prob,no_solutions)
pop.x=repmat(prob.bounds(:,1)',no_solutions,1)+repmat((prob.bounds(:,2)-prob.bounds(:,1))',no_solutions,1).*rand(no_solutions,prob.nx);
return