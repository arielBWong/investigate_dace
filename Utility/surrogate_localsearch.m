function[newx] = surrogate_localsearch(newx, prob, krg_obj, krg_con, daceflag)
%----------------------------

if length(krg_obj) > 1
    % no local search for MO
     return
end

 funh_obj = @(x)surrogate_predict(x, krg_obj, daceflag);
 funh_con = @(x)surrogate_predict(x, krg_con, daceflag);

opts = optimset('fmincon');
opts.Algorithm = 'sqp';
opts.Display = 'off';
opts.MaxFunctionEvaluations = 100;
[newx, newf, ~, output] = fmincon(funh_obj, newx, [], [],[], [],  ...
    prob.xl_bl, prob.xl_bu, [],opts);

end
