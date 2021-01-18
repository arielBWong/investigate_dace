function x=Run_EA(prob,param,s_pop,archive,gprMdl)
for i=1:param.EA_generations
    % Generate offspring
    c_pop.x=Generate_child(s_pop,prob,param);
    
    % Approximate these offspring
    [mu,sigma]=Predict_GPR(gprMdl,c_pop.x,param,archive);
    c_pop.muf=mu(:,1:prob.nf);
    c_pop.sf=sigma(:,1:prob.nf);
    if(prob.ng>0)
        c_pop.mug=mu(:,prob.nf+1:end);
        c_pop.sg=sigma(:,prob.nf+1:end);
    end
    
    % Selecting the parent pop for next generation
    [s_pop]=Reduce(prob,param,c_pop,s_pop,archive);
end

% Identifying the best solution
x=Identify_best(archive,s_pop,param,prob);

if(param.local_search==1)
    % Conduct a local search to improve this x
    x=Local_search(x,param,prob,archive,gprMdl);
end

return
