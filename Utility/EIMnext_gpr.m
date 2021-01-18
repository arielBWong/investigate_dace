function[best_x, info] = EIMnext_gpr(train_x, train_y, xu_bound, xl_bound, ...
                                 num_pop, num_gen, train_c, fitnesshn, normhn, varargin)
% method of using EIM to generate next point
% usage:
%
% input: 
%         train_x                   : design variables
%                                               1/2d array: (num_samples, num_varibles)
%        train_y                    : objective values
%                                               1/2d array: (num_samples, num_objectives)
%        xu_bound                   : upper bound of train_x
%                                               1d array 
%        xl_bound                   :  lower bound of train_x
%                                               1d array
%        num_pop                    : EIM optimization parameter
%        num_gen                    : EIM optimization parameter
%        train_c                    : constraints values
%                                               1/2d array: (num_samples, num_constraints)
% output: 
%        best_x                     : proposed next x to be evaluated by EIM
%         info                      : returned information for functor caller to recreate 
%                                           or check information
%                                           info.krg  
%                                           info.krgc
%                                           info.train_xmean
%                                           info.train_ymean
%                                           info.train_xstd
%                                           info.train_ystd
%                                           info.info.eim_normf
%--------------------------------------------------------------------------

% number of objective
num_obj = size(train_y, 2);
kriging_obj = cell(1,num_obj);

% number of input designs
num_x = size(train_x,1);
num_vari = size(train_x, 2);

% test purpose---
ub = xu_bound;
lb = xl_bound;
train_x_norm = train_x;
x_mean = NaN;
x_std = NaN;
%-------------

% train objective kriging model
if num_obj > 1
    train_y_norm = normhn(train_y);
    y_mean = NaN;
    y_std = NaN;

else
   [train_y_norm, y_mean, y_std] = zscore(train_y); 
end

for ii = 1:num_obj        
        kriging_obj{ii} = myfitgrp(train_x_norm,train_y_norm(:,ii),5);
end

% prepare f_best for EIM, first only consider non-constraint situation
if num_obj > 1
    index_p = Paretoset(train_y_norm);
    f_best = train_y_norm(index_p, :); % constraint problem has further process
else
    f_best = min(train_y_norm, [], 1);
end

% compatibility with constraint problems
if ~isempty(train_c)
    num_con = size(train_c, 2);
    kriging_con = cell(1,num_con);
    train_c_norm = train_c;
    
    for ii = 1:num_con
        kriging_con{ii} = myfitgrp(train_x_norm,train_c_norm(:,ii), 5);
    end
    % adjust f_best according to feasibility
    % feasibility needs to be valued in original range
    index_c = sum(train_c <= 0, 2) == num_con;
    if sum(index_c) == 0 % no feasible
        f_best = [];
    else
        feasible_trainy_norm = train_y_norm(index_c, :);
        % still needs nd front
        index_p = Paretoset(feasible_trainy_norm);
        f_best = feasible_trainy_norm(index_p, :);
    end   
    fitness_val = @(x)fitnesshn(x,f_best, kriging_obj, kriging_con); % paper version
else
    fitness_val = @(x)fitnesshn(x, f_best, kriging_obj);
end

%----------
% fighn = varargin{1};

% call evolution
funh_con = @(x) con(x);
param.gen = num_gen;
param.popsize = num_pop;

[best_x, eim_f, bestc, archive] = gsolver(fitness_val, num_vari,  lb,ub, [], funh_con, param);

% ---add local search on EI
if isempty(train_c)
    best_x = EI_localsearch(best_x, eim_f, lb, ub, kriging_obj);
else
    best_x = EI_localsearch(best_x, eim_f, lb, ub, kriging_obj, kriging_con);
end



% ---add local search
info = struct();
info.eim_normf = eim_f;
info.krg = kriging_obj;
info.train_xmean = x_mean;
info.train_xstd = x_std;
info.train_ymean = y_mean;
info.train_ystd = y_std;
info.stable = true;

if  ~isempty(train_c)
    info.krgc = kriging_con;
else
    info.krgc = [];
end

end

function c = con(x)
c=[];
end




function [newx] = EI_localsearch(newx, fbest, bl, bu, kriging_obj, kriging_con)
% local search on EI function
%--------------
num_vari = size(newx, 2);
if  nargin > 5
    funh_obj = @(x)EIM_eval(x, fbest, kriging_obj, kriging_con);
else
    funh_obj = @(x)EIM_eval(x, fbest, kriging_obj);
end

opts = optimset('fmincon');
opts.Algorithm = 'sqp';
opts.Display = 'off';
opts.MaxFunctionEvaluations = 100;
[newx_local, newf, ~, output] = fmincon(funh_obj, newx, [], [], [], [],  ...
    bl, bu, [], opts);

if newf <= fbest % local search got problem
    newx = newx_local;
end
    

end
