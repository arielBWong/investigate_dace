function [fit] = EIM_evaldaceUpdate(x, f, arc_obj, kriging_obj, arc_con, kriging_con)
% function of using EIM as fitness evaluation
%--------------------------------------------------------------------------

% number of input designs
num_x   = size(x, 1);
num_obj = size(f, 2);

% the kriging prediction and varince
param.GPR_type   = 2;
param.no_trials  = 5;

% pof init
pof = 1;

if isempty(f) && nargin > 3   % refer to no feasible solution
    % only calculate probability of feasibility   
    % for constraint there is no normalization, so no problem 
    [mu_g, mse_g]= Predict_GPR(kriging_con, x, param, arc_con);
 
    pof          = prod(Gaussian_CDF((0 - mu_g)./mse_g), 2);
    fit          = -pof;
    return
end

% predict objective value
[u, mse]        = Predict_GPR(kriging_obj, x, param, arc_obj);

% calcualate eim for objective
if num_obj == 1
    f          = repmat(f, num_x, 1);
    imp        = f - u;
    z          = imp ./ mse;
    ei1        = imp .* Gaussian_CDF(z);
    ei1(mse==0)= 0;
    ei2        = mse .* Gaussian_PDF(z);
    EIM        = (ei1 + ei2);
else
    % for multiple objective problems
    % f - refers to pareto front
    r               = 1.1*ones(1, num_obj);  % reference point
    num_pareto      = size(f, 1);
    r_matrix        = repmat(r,num_pareto,1);
    EIM             = zeros(num_x,1);
    for ii = 1 : num_x
        u_matrix    = repmat(u(ii,:),num_pareto,1);
        s_matrix    = repmat(mse(ii,:),num_pareto,1);
        eim_single  = (f - u_matrix).*Gaussian_CDF((f - u_matrix)./s_matrix) + s_matrix.*Gaussian_PDF((f - u_matrix)./s_matrix);
        EIM(ii)     =  min(prod(r_matrix - f + eim_single,2) - prod(r_matrix - f,2));
    end
end

% calculate probability of feasibility for constraint problem
if nargin > 4  
    % the kriging prediction and varince
    [mu_g, mse_g] = Predict_GPR(kriging_con, x, param, arc_con);
     pof          = prod(Gaussian_CDF((0-mu_g)./mse_g), 2);
end
fit = -EIM .* pof;
end

