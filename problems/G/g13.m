function [f,g] = g13(x)
if nargin == 0
	prob.nx = 5;
	prob.nf = 1;
	prob.ng = 3;
	prob.range = cell(1, prob.nx);
	for i = 1:prob.nx
		prob.range{i} = Range('range', [-3.2,3.2]);
	end
	f = prob;
else
	[f,g] = g13_true(double(x));
end
return


function [f,g] = g13_true(x)
f = exp(prod(x));

tmp1 = x*x' - 10;
tmp2 = x(2)*x(3) - 5*x(4)*x(5);
tmp3 = x(1)^3 + x(2)^3 + 1;

eps = 5.e-4;

g(1) = eps - abs(tmp1);
g(2) = eps - abs(tmp2);
g(3) = eps - abs(tmp3);

% g(1) = tmp1 + eps;
% g(2) = eps - tmp1;
% g(3) = tmp2 + eps;
% g(4) = eps - tmp2;
% g(5) = tmp3 + eps;
% g(6) = eps - tmp3;
return
