function c = constraint2_l(xu, xl)
% this constraint is active 
% stops xl2 to approach the mimimum value
% consequently the pareto front of upper level problem 
% is moved upwards
 c = (xl(:, 2) - xu(:, 2)).^2 - 1;
 c = -c;
 
end