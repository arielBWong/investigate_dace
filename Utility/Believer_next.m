function[best_x, info] = Believer_next(train_x, train_y, xu_bound, xl_bound, ...
                                 num_pop, num_gen, train_c, fitnesshn, normhn, varargin)
   % fitness is an useless position, but cannot be deleted at this stage
                                                          
    daceflag = varargin{1};
    prob =  varargin{2};
    
    train_y_norm = normhn(train_y);
    krg_obj = surrogate_train(train_x, train_y_norm, daceflag);
    krg_con = surrogate_train(train_x, train_c, daceflag);
                              
    funh_obj = @(x)surrogate_predict(x, krg_obj, daceflag);
    funh_con = @(x)surrogate_predict(x, krg_con, daceflag);
    
    param.gen=num_gen;
    param.popsize = num_pop;
    
    l_nvar = size(train_x, 2);
    [~,~,~, archive] = gsolver(funh_obj, l_nvar,  xl_bound, xu_bound, [], funh_con, param);    
    [newx, growflag] = believer_select(archive.pop_last.X, train_x, prob, false, true);
    
    if growflag % there is unseen data in evolution
        [best_x] = surrogate_localsearch(newx, prob, krg_obj, krg_con, daceflag); 
        % inprocess_plotsearch(fighn, prob, cons_hn, new_xl, train_xl);

    else % there is no unseen data in evolution
        % re-introduce random individual
        fprintf('In believer, no unseen data in last population, introduce randomness \n');
        best_x = [];
    end

    %-----stability
    info = struct();
    info.krg = krg_obj;
    if  ~isempty(train_c)
        info.krgc = krg_con;
    else
        info.krgc = [];
    end
    if ~daceflag
        info.stable = true;
    else
        info.stable = stability_check(train_x, train_y, train_c, krg_obj, krg_con);
    end
end



