function dmodel = Train_DACE(X,Y,no_trials)


% Check design points
[m1, nx] = size(X);
[m2, ~]  = size(Y);
if m1 ~= m2
	error('X and Y must have the same number of rows')
end
range = minmax(X');
lb    = max(1e-6,range(:,1));
ub    = range(:,2);

fmin=Inf;
for trials = 1:no_trials
    theta_start  = lb+(ub-lb).*rand(nx,1);
    dmodel       = Fit_DACE(X, Y, @Regpoly0, @Corrgauss, theta_start', lb, ub);
    samples      = size(X, 1);
    %----- prediction
    if samples > 1
        [f, ssqr]    = Predictor(X, dmodel);
    else
        [f, ~, ssqr] = Predictor(X, dmodel);
    end
    if size(f,1)     == size(X,1)
        y = f;
    else
        y = f';
    end
    sigma = sqrt(abs(ssqr));
    y_prd = y;
    y_sig = sigma;
    %------------------
    
    fpr   = sqrt(sum((y_prd - Y).^2));    
    if(fpr<=fmin)
        output_model=dmodel;
        fmin=fpr;
    end
end

dmodel = output_model;
return