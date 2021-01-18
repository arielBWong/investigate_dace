%% development test on gsolver
% comparision is conducted with matlab ga in unitest_comparewith_gasolver
clearvars;
workdir = pwd;
idcs = strfind(workdir, '\');
upperfolder = workdir(1: idcs(end)-1);
problem_folder = strcat(upperfolder,'\problems\SMD');
addpath(problem_folder);

prob = smd11();
xu = [0, 0];

%-global search
obj = @(x)hyobj(x, xu, prob);
con = @(x)hycon(x, xu, prob);

param.popsize = 50;
param.gen = 100;
initxl = [];

 [bestx, bestf, bestc]  = gsolver(obj,prob.n_lvar, prob.xl_bl, prob.xl_bu, initxl, con, param);

 c = hycon(bestx, xu, prob);
 
rmpath(problem_folder)

function [f] = hyobj(x, xu, prob)
    [f, ~] = prob.evaluate_l(xu, x);
end

function [c] = hycon(x, xu, prob)
    [~, c] = prob.evaluate_l(xu, x);

end