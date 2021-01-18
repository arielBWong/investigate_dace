function [f,g] = g11(x)
if nargin == 0
	prob.nx = 2;
	prob.nf = 1;
	prob.ng = 2;
	prob.range = {Range('range', [-1,1]), Range('range', [-1,1])};
	f = prob;
else
	[f,g] = g11_true(double(x));
end
return


function [f,g] = g11_true(x)
f = x(1)^2 + (x(2)-1)^2;
tmp1 = x(2) - x(1)^2;
eps = 1.e-3;
g(1) = tmp1 + eps;
g(2) = eps - tmp1;
return
