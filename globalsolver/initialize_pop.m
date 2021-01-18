function [pop,archive]=initialize_pop(funh_obj, funh_con, num_xvar, xl, xu, initmatrix, param)
% funh_obj, funh_con, num_xvar, ops, evals, xu, xl
% this method generate the initial population of global solver
%
%   input
%       funh_obj                                                : function handle for objective function
%       funh_con                                               : function handle to contraints
%       num_xvar                                              : number of design variables
%       xu                                                            : 1d row vector, upper bound of design variables
%       xl                                                             : 1d row vector, lower bound of design variables
%       initmatrix                                              :  partial population inserted to evolution, put in [] if no partial
%                                                                          population is given
%       param                                                    : evolutionary parameters (popsize, gen)
%   output
%       pop                                                        : population structure with fields: X(design variable),
%                                                                                    F(objective), C(constraint)
%       archive                                                   : record  all solutions encountered over evolution
%                                                                                      * usage under development
%---------------------------------------------------------------------------------
N = param.popsize;

% create initial population,  w.r.t.  given partial population intimatrix
n_init = size(initmatrix, 1);
n_rest = N - n_init;
X_pop = repmat(xl, n_rest, 1) + repmat(xu - xl, n_rest, 1) .* lhsdesign(n_rest, num_xvar);
X_pop = [X_pop; initmatrix];

% objective and constraints
F_pop=funh_obj(X_pop);
C_pop = funh_con(X_pop);

[~,ids,~] = nd_sort(F_pop, (1:size(F_pop,1))');

% Storing relevant information pop and archive
pop.X=X_pop(ids,:);
pop.F=F_pop(ids,:);

if isempty(C_pop)
    pop.C = [];
else
    pop.C = C_pop(ids, :);
end

archive.sols=[repmat(0,N,1),  pop.X, pop.F, pop.C];

return

