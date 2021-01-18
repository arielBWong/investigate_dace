function c = constraint2_u(xu)
% this constaint is to seperate the upper level pareto front 
% xu1 takes range [0, k], xu2 takes range [-k, k] 
% the following constraints on upper level makes half of upper pf 
% feasible
% when xu2 is at best value of xu2 is 0.5-> (2-1)/2
% xu1 can only form half of the pareto front
% if xu2 is not at its best
% xu1 
c = xu(:, 1) - (0.5 - xu(:, 2).^2);
end
