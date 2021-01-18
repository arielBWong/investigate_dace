function [fit] = EIM_eval(x, f, kriging_obj, kriging_con)
% function of using EIM as fitness evaluation
% usage:
%
% input: 
%        x                         - pop to evaluate
%        f                         - best f so far/feasible pareto front
%                                           in multi-objective probs
%        kriging_obj               - kriging model for objectives
%        kriging_con               - kriging model for constraints
% output: 
%       fit                         - pop fitness
%--------------------------------------------------------------------------


% number of input designs
num_x = size(x,1);
num_obj = size(f, 2);

% the kriging prediction and varince
u = zeros(num_x,num_obj);
mse = zeros(num_x,num_obj);

% pof init
pof = 1;

if isempty(f) && nargin > 3   % refer to no feasible solution
    % only calculate probability of feasibility
    num_con = length(kriging_con);
    % the kriging prediction and varince
    mu_g = zeros(size(x,1), num_con);
    mse_g = zeros(size(x,1), num_con);
    for ii = 1: num_con
        [mu_g(:, ii), mse_g(:, ii)] = predict(kriging_con{ii}, x);
    end
    pof  = prod(Gaussian_CDF((0-mu_g)./mse_g), 2);
    fit = -pof;
    return
end

for ii = 1:num_obj
    [u(:, ii), mse(:, ii)] = predict(kriging_obj{ii}, x);
end

% calcualate eim for objective
if num_obj == 1
    f = repmat(f, num_x, 1);
    imp = f - u;
    z = imp./mse;
    ei1 = imp .* Gaussian_CDF(z);
    ei1(mse==0)=0;
    ei2 = mse .* Gaussian_PDF(z);
    EIM = (ei1 + ei2);
else
    % for multiple objective problems
    % f - refers to pareto front
    r = 1.1*ones(1, num_obj);  % reference point
    num_pareto = size(f, 1);
    r_matrix = repmat(r,num_pareto,1);
    EIM = zeros(num_x,1);
    for ii = 1 : num_x
        u_matrix = repmat(u(ii,:),num_pareto,1);
        s_matrix = repmat(mse(ii,:),num_pareto,1);
        eim_single = (f - u_matrix).*Gaussian_CDF((f - u_matrix)./s_matrix) + s_matrix.*Gaussian_PDF((f - u_matrix)./s_matrix);
        EIM(ii) =  min(prod(r_matrix - f + eim_single,2) - prod(r_matrix - f,2));
    end
end

% calculate probability of feasibility for constraint problem
if nargin>3
    % the number of constraints
    num_con = length(kriging_con);
    % the kriging prediction and varince
    mu_g = zeros(size(x,1), num_con);
    mse_g = zeros(size(x,1), num_con);
    for ii = 1: num_con
        [mu_g(:, ii), mse_g(:, ii)] = predict(kriging_con{ii}, x);
    end
    pof  = prod(Gaussian_CDF((0-mu_g)./mse_g), 2);
end
fit = -EIM .* pof;
end
