function[best_x, info] = switchBeliever_gpr(train_x, krg_obj, krg_con, ...
                                 num_pop, num_gen, varargin)
% fitness is an useless position, but cannot be deleted at this stage
                                                          
    daceflag = varargin{1};
    prob =  varargin{2};

                              
    funh_obj = @(x)surrogate_predict(x, krg_obj, daceflag);
    funh_con = @(x)surrogate_predict(x, krg_con, daceflag);
    
    param.gen=num_gen;
    param.popsize = num_pop;
    
    l_nvar = size(train_x, 2);
    [~,~,~, archive] = gsolver(funh_obj, l_nvar,  prob.xl_bl, prob.xl_bu, [], funh_con, param);    
    [newx, growflag] = believer_select(archive.pop_last.X, train_x, prob, false, true);
    
    if growflag % there is unseen data in evolution
        [best_x] = surrogate_localsearch(newx, prob, krg_obj, krg_con, daceflag); 
        % inprocess_plotsearch(fighn, prob, cons_hn, new_xl, train_xl);

    else % there is no unseen data in evolution
        % re-introduce random individual
        fprintf('In believer, no unseen data in last population, introduce randomness \n');
        best_x = [];
    end

    
    info = [];
end





