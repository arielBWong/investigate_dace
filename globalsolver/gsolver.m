function [bestx, bestf, bestc, archive] = gsolver(funh_obj, num_xvar, lb, ub, initmatrix, funh_con, param, varargin)
% This function is a wrapper on methods in global optimization/minimization
% folder. Main process is nsga, but reproducation is a DE operator
% Be aware, this method only handle inequality constraints
% Don't use on equality constraints
% input:
%       funh_obj : function handle to objective function
%       num_xvar : number design variables
%       lb: upper bound of design varibles
%                1d row array
%       up: lower bound of design variables
%       initmatrix:  partial population to be embeded in
%                           initial population
%       funh_con : function handle to constraint functions
%       param : structure specifying ea parameters(param. popsize; param.gen)
%       varargin : additional variables for dealing with bilevel problems
% output:
%       bestx : global search results of design variables  (best value or nd front)
%       bestf : global search results of objective values   (best value or nd front)
%       bestc : global search results of constraints
%                       for constraint problems, if no feasible is found, return least infeasible one
%  ** under development of archive handling
%--------------------------------------------------------------------------

bestx = NaN;
bestf = NaN;
bestc = NaN;


% Initialization
[pop,archive] = initialize_pop(funh_obj, funh_con, num_xvar, lb, ub, initmatrix, param);

%%%%%%%%%%%%%%%%%%%%%
if ~isempty(varargin)
    f1 = figure(1);
    lb1 = lb(1);
    ub1 = ub(1);
    
    lb2 = lb(2);
    ub2 = ub(2);
    num_points = 101;
    x1 = linspace(lb1, ub1, num_points);
    x2 = linspace(lb2, ub2, num_points);
    [x1, x2] = meshgrid(x1, x2);
    testdata = zeros(num_points, num_points);
    
    for i = 1:num_points
        for j = 1:num_points
            [f(i, j), ~] = funh_obj([x1(i, j), x2(i, j)]);
        end
    end
    surf(x1, x2, f); hold on;
    colormap parula
    shading interp
    scatter3(pop.X(:, 1), pop.X(:, 2), pop.F, 50, 'filled', 'r');
end
%%%%%%%%%%%%%%%%%%%%

gen=1;
while gen<= param.gen
    % Recombination
    child.X=generate_child(lb, ub, pop, param);
    
    % Evaluate and Order(need to test varargin)
    [pop,archive]= evaluate_order(pop,archive, funh_obj, funh_con, child.X, gen, param);
    
    % Reduce 2N to N
    [pop]=reduce_pop(pop,param.popsize);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    if ~isempty(varargin)
        clf(f1);
        surf(x1, x2, f); hold on;
        colormap parula
        shading interp
        scatter3(pop.X(:, 1), pop.X(:, 2), pop.F, 50, 'filled', 'r');
        pause(0.5);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%
    gen = gen+1;
    % disp(gen);
end



if ~isempty(varargin)
    close(f1);  
end


% use archive to save last pop_x
archive.pop_last =pop;

num_obj = size(pop.F, 2);
num_con = size(pop.C, 2);

if num_obj == 1
    bestx = pop.X(1, :);
    bestf = pop.F(1, :);
    if ~isempty(pop.C)
        bestc = pop.C(1, :);
    else
        bestc = [];
    end
    return
end

% return feasible nd front
if num_obj > 1
    if ~isempty(pop.C)                                    % constraint problem
        pop.C(pop.C<=0) = 0;
        fy_ind = sum(pop.C, 2) ==0;
    else
        fy_ind =true(param.popsize ,1);                    % unconstraint problem
    end
end

% feasible exists for mo problem
if sum(fy_ind) > 0
    [fronts, ~, ~] = nd_sort(pop.F, find(fy_ind));
    bestf = pop.F(fronts(1).f, :);
    bestx = pop.X(fronts(1).f, :);
    if ~isempty(pop.C)
        bestc = pop.C(fronts(1).f, :);
    else
        bestc = [];
    end
else
    % no feasible solution in final population
    % return least infeasible on
    sumc= sum(pop.C, 2);
    [~, id] = sort(sumc);
    bestf = pop.F(id(1),  :);
    bestx = pop.X(id(1),  :);
    bestc = pop.C(id(1),  :);
end
end
