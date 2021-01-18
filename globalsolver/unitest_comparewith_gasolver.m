%% development test on gsolver
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

opts = optimoptions('ga');
opts.MaxGenerations = 100;
opts.PopulationSize = 50;

[newxl_g, newfl_g] = ga(obj, prob.n_lvar, [],[],[],[], prob.xl_bl, prob.xl_bu,con, opts);

[newxc, ~] = hycon(newxl_g, xu, prob);

rmpath(problem_folder)

function [f] = hyobj(x, xu, prob)
    [f, ~] = prob.evaluate_l(xu, x);
end

function [c, ceq] = hycon(x, xu, prob)
    [~, c] = prob.evaluate_l(xu, x);
    ceq = [];
end