function c = constraint1_l(xu, xl)
% this constraint puts both upper level and lower level 
% variables together 
% this contraints only set feasible around optimum
 c = (xl(:, 1) - xu(:, 1)).^2 + (xl(:, 2) - xu(:, 2)).^2 - 2;
 
end