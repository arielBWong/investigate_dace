function[match_xl, n_fev, flag] = ego_EIdace(xu, prob, seed)
% method of searching for a match xl for xu.
% Problem(Prob) definition require certain formation for bilevel problems
% evaluation method should be of  form 'evaluation_l(xu, xl)'
% usage
% input:
%        xu:  upper level variable, to be matched
%        prob: problem instance, require certin evaluation
%                      method name defintion-- check problems
%                      folder
%        num_pop : DE parameter
%        num_gen : DE parameter
%        propose_nextx  : str, function handle to generate next x
%        iter_size : surrogate parameter: number of iterations
%        llfit_hn :  str, lower level seach fitness for next x method, used
%                          by proposed_nextx method
%
% output:
%        matching_xl : found xl for xu
%        n_fev : total number of function evaluation on lower level
%        flag : whether xl is a feasible solution(true/false)
%--------------------------------------------------------------------------

parameterfile = strcat(pwd, '\parameter_ei.m');
run(parameterfile)
load('param_ei');
rng(seed, 'twister');

num_pop = param_ei.num_pop;
num_gen = param_ei.num_gen;
init_size = param_ei.init_size;
iter_size = param_ei.iter_size; 
propose_nextx= param_ei.propose_nextx;
norm_str = param_ei. norm_str;
llfit_hn = param_ei.llfit_hn;



prob = eval(prob);
l_nvar = prob.n_lvar;
% init_size = 11 * l_nvar - 1;
% init_size = 7;

upper_bound = prob.xl_bu;
lower_bound = prob.xl_bl;
xu_init = repmat(xu, init_size, 1);
train_xl = lhsdesign(init_size,l_nvar,'criterion','maximin','iterations',1000);
train_xl = repmat(lower_bound, init_size, 1) ...
    + repmat((upper_bound - lower_bound), init_size, 1) .* train_xl;


% ---demo test
initx = train_xl;
% initx = [0, 0.5, 1]';
% ---demo test

% evaluate/get training fl from xu_init and train_xl
% compatible with non-constriant problem
[train_fl, train_fc] = prob.evaluate_l(xu_init, train_xl);



fighn2 = figure(2);

% call EIM/Ehv to expand train xl one by one
nextx_hn = str2func('EIMnext');
fithn = str2func('EIM_evaldace');

normhn = str2func(norm_str);
daceflag = true;
for iter = 1:iter_size
    fprintf('iteration %d: \n', iter);
    % eim propose next xl
    % lower level is single objective so no normalization method is needed
    

    [new_xl, infor] = nextx_hn(train_xl, train_fl, upper_bound, lower_bound, ...
        num_pop, num_gen, train_fc, fithn, normhn, fighn2);
  
    
    if ~infor.stable
        fprintf('not stable at iteration %d\n', iter);

    end
    
    before = new_xl;
    
    % local search on surrogate
    % evaluate next xl with xu
    [new_fl, new_fc] = prob.evaluate_l(xu, new_xl);

    fprintf('dace flag is %d \n', daceflag);
    krg_obj = infor.krg;
    if size(train_xl, 2)==1
        processplot1d(fighn2, train_xl, train_fl, krg_obj, prob, initx, before, daceflag);
    end
    %--forrest demo
    
    % inprocess_plotsearch(fighn, prob, cons_hn, new_xl, train_xl);
    % add to training
    
    %--- closeness check---
    check = abs(train_xl - new_xl);
    check = round(check, 5);
    if length(unique(check, 'rows')) < size(train_xl, 1)
        fprintf('fail unique check\n');
        disp(new_xl);
        continue;
    end
    
    train_xl = [train_xl; new_xl];
    train_fl = [train_fl; new_fl];
    train_fc = [train_fc; new_fc];  % compatible with nonconstraint
  
end


[best_x, best_f, best_c, s] =  localsolver_startselection(train_xl, train_fl, train_fc);
nolocalsearch = true;
if nolocalsearch
    match_xl = best_x;
    n_fev = size(train_xl, 1);
    flag = s;
else
     if size(train_fl, 2)> 1
        error('local search does not apply to MO');
    end
    [match_xl, flag, num_eval] = ll_localsearch(best_x, best_f, best_c, s, xu, prob);
    n_global = size(train_xl, 1);
    n_fev = n_global +num_eval;       % one in a population is evaluated
end

% save lower level
llcmp = true;
% llcmp = false;
if llcmp
    method = 'lleim';

       % add local search result
    % only for SO
    if size(train_fl, 2) ==  1
        train_xl = [train_xl; match_xl];
        [local_fl, local_fc]  = prob.evaluate_l(xu, match_xl);
        train_fl = [train_fl; local_fl];
        train_fc = [train_fc; local_fc];
    end
    perfrecord_umoc(xu, train_xl, train_fl, train_fc, prob, seed, method, 0, 0, init_size);
        
end
end



function tooclose = archive_check(newx, trainx, prob)
% ---check newx whether it is
tooclose = false;
eps_dist = sqrt(prob.n_lvar) * 0.01;

upper_bound = prob.xl_bu;
lower_bound = prob.xl_bl;

trainx_norm = (trainx - lower_bound) ./ (upper_bound - lower_bound);
newx_norm = (newx - lower_bound) ./ (upper_bound - lower_bound);

%---
mindistance = min(pdist2(newx_norm,trainx_norm));

if mindistance < eps_dist
    tooclose =  true;
end
end



function  [f, sig] = llobj(x, kriging_obj, daceflag)
num_obj = length(kriging_obj);   % krg cell array?
num_x = size(x, 1);
f = zeros(num_x, num_obj);
for ii =1:num_obj
    if daceflag
        [f(:, ii), sig] = predictor(x, kriging_obj{ii});
        
    else
        [f(:, ii), sig] =  predict(kriging_obj{ii}, x);
    end
end
end

% ---demo test: forrestor
function[] = processplot(fighn, trainx, trainy, krg, prob, initx, before, daceflag)

crosscheck(krg{1}, trainx, trainy, daceflag);
clf(fighn);
% (1) create test
testdata = linspace(prob.xl_bl, prob.xl_bu, 100);
testdata = testdata';

% (2) predict
[fpred, sig] = llobj(testdata, krg, daceflag);


yyaxis left;
fpred = denormzscore(trainy, fpred);
plot(testdata, fpred, 'r--', 'LineWidth', 1); hold on;

% sig variation
y1 = fpred + sig * 10;
y2 = fpred - sig * 10;
y = [y1', fliplr(y2')];
x = [testdata', fliplr(testdata')];
fill(x, y, 'r', 'FaceAlpha', 0.1, 'EdgeColor','none'); hold on;

% (3) real
[freal, sig]= prob.evaluate_l([], testdata);
plot(testdata, freal, 'b', 'LineWidth', 2);hold on;

% (4) scatter train
scatter(trainx, trainy, 80, 'ko', 'LineWidth', 2);

inity = prob.evaluate_l([], initx);
scatter(initx, inity, 40, 'ro', 'filled');
%---
newy = prob.evaluate_l([], before);
scatter(before, newy, 80, 'ko', 'filled');

% (5) calculate EI and plot
yyaxis right;
[train_ynorm, ~, ~] = zscore(trainy);

ynorm_min = min(train_ynorm);
fprintf('rebuild landscape best f: %f\n',ynorm_min );
if daceflag
    fit = EIM_evaldace(testdata, ynorm_min,  krg, []);
else
    fit = EIM_eval(testdata, ynorm_min,  krg, []);
end
fit = -fit;
plot(testdata, fit, 'g--');

data = [trainx, trainy];
save('data', 'data');
pause(0.5);
end

function crosscheck(krg, trainx, trainy, daceflag)
if daceflag
    [yn, ~] = predictor(trainx, krg);
else
    [yn, ~] = predict(krg, trainx);
end

y =  denormzscore(trainy, yn);
b = max(abs(y - trainy)) - 0.01;
disp(b);
fprintf('stability check should be negative: %f\n',b);
% save('sig_dace', 'sig_dace');

a = unique(round(trainx, 3));
% size(a)
% size(trainx)
end

 function f = denormzscore(trainy, fnorm)
[train_y_norm, y_mean, y_std] = zscore(trainy);
f = fnorm * y_std + y_mean;
 end
