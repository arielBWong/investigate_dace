function [pop, archive]= evaluate_order(pop, archive, funh_obj, funh_con, cx, gen, param, varargin)
% This function evaluate given x population, add to pop (N to 2N)
% and sort pop with nd and feasibility consideration
% input
%           pop : population of previous generation
%           archive : record of all evolutionary process
%           funh_obj : function handle of objective function
%           funh_con: function handle of constraints
%           cx: child population generated from last step
%           gen: current generation
%           param: evolution parameter (gen popsize)
% output
%           pop: extended and sorted population
%           archive: extended archive
%-------------------------
child.X=cx;
% This is NSGA-II
child.F = funh_obj(child.X);
child.C = funh_con(child.X);
numcon = size(child.C, 2);

archive.sols=[archive.sols;[repmat(gen,param.popsize,1), child.X, child.F, child.C]];

% Appending X F C to pop
pop.X=[pop.X;child.X];
pop.F=[pop.F;child.F];
pop.C = [pop.C; child.C];

[pop.F, pop.X, pop.C] = pop_sort(pop.F, pop.X, pop.C);

% dealing with feasibility on lower level
if ~isempty(varargin)
    prob = varargin{1};
    xu_g =  varargin{2};
    xl_g = varargin{3};
    pop.F = pop_llfeasicheck(pop.X, pop.F, xu_g, xl_g, prob);
end

end

function[mf] = pop_llfeasicheck(xu, fu, xu_g, xl_g, probh)
% this function considers bilevel situation, where lower level is infeasible
%  it collects all upper level functions that get evaluated (it is infact all past population)
% uses its highest values (minimization problem)  to punish ll infeasible
% solutions
% usage
%   input
%  xu : current population
%  fu:  current population objectives to be re-accessed
%  xu_g : global variable passed from outside
%  xl_g :  global variable passed from outside
% probh: problem instance
%   output
%   : modified current population on fu
% * test required
%-----------------------------------------------------------------

% get feasibility from global saves
% since global xu xl are already evaluated
% here calculating fc is not counted for algorithm clearity in 
% compatible with addons for gslover, sacrifices efficiency for
% compatibility
workdir = pwd;
idcs = strfind(workdir, '\');
upperfolder = workdir(1: idcs(end)-1);
addpath(upperfolder);   % llfeasi_modify exists in upper folder

num_pop = size(fu, 1);
xl = [];
for i = 1:num_pop
    match =  check_exist(xu(i, :), xu_g, xl_g);
    if isempty(xl)
        error('logically wrong, by far, every xu in population should have xl in global variable');
    end
    xl = [xl; match];  
end

[~, fc] = probh.evaluate_l(xu, xl);
if ~isempty(fc)
    % in order to reuse function llfeasi_modify
    % construct arguments for llfeasi_modify
    [~, fc_g]  = probh.evaluate_l(xu_g, xl_g);
    [fu_g, ~] = probh.evaluate_u(xu_g, xl_g);
    num_con = size(fc_g, 2);
    num_g = size(fc_g, 1);
    % (1) feasi_flag of all solutions
    fc_all = [fc_g; fc];
    fc_all(fc_all<=0) = 0;
    feasi_flag = sum(fc_all, 2)==0;
    % (2) contruct all current evaluated fus
    fu_all = [fu_g; fu];
    % (3) process fu one by one use index in all solutions
    for ii = num_g + 1: num_g + num_pop
        fu_all = llfeasi_modify(fu_all, feasi_flag, ii);
    end
    mf = fu_all(num_g + 1: num_g + num_pop, :);
    
else
    mf = fu;
end
%----------------------------------------------
end


function xl = check_exist(xu, xu_g, xl_g)
%  this function should can be extracted
% as it is also used in ul_llea m file
%--------------------------------------
xu = round(xu, 10);
if isempty(xu_g)
    xl = [];
else
    xu_g = round(xu_g, 10);
    diff = xu_g - xu;
    ind = sum(diff, 2) == 0;
    
    if sum(ind)>0           % should be only 1, if exists
        % fprintf('found xu in global save %d\n', sum(ind));
        xl = xl_g(ind, :);
        if sum(ind)>1
            error('there cannot be repeat x more than once')
        end
    else
        xl = [];
    end
end
end

