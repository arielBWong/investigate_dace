function [f,g] = g3(x)
if nargin == 0
	prob.nx = 10;
	prob.nf = 1;
	prob.ng = 2;
	prob.range = cell(1, prob.nx);
	for i = 1:prob.nx
		prob.range{i} = Range('range', [0,1]);
	end
	f = prob;
else
	[f,g] = g3_true(double(x));
end
return


function [f,g] = g3_true(x)
n = length(x);
f = - (sqrt(n))^n * prod(x);

tmp1 = x*x';
eps = 1.e-3;
g(1) = tmp1-1 + eps;
g(2) = eps - (tmp1-1);
return
