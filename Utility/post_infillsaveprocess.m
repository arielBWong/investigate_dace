function [match_xl, n_fev, flag] = post_infillsaveprocess(xu, train_xl, train_fl, train_fc, str_method, seed, prob, init_size)
% local search for best xl
% connect a local search to sao
% local search starting point selection
% MO problem no process
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


%----
% save lower level
llcmp = true;
%llcmp = false;
if llcmp
    method = str_method;
  
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