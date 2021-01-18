function [y_prd,y_sig] = Predict_DACE(gprMdl,x_tst,id)
id_response=id;
X=x_tst;
dmodel=gprMdl;
samples = size(X,1);
if samples > 1
    [f, ssqr] = Predictor(X, dmodel);
else 
    [f, ~, ssqr] = Predictor(X, dmodel);
end
if size(f,1) == size(X,1)
	y = f;
else
	y = f';
end
sigma = sqrt(abs(ssqr));
y_prd=y;
y_sig=sigma;
return
