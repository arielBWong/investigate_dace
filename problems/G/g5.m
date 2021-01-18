function [f,g] = g5(x)
if nargin == 0
	prob.nx = 4;
	prob.nf = 1;
	prob.ng = 8;
	prob.range = cell(1, prob.nx);
	prob.range{1} = Range('range', [0,1200]);
	prob.range{2} = Range('range', [0,1200]);
	prob.range{3} = Range('range', [-0.55,0.55]);
	prob.range{4} = Range('range', [-0.55,0.55]);
	f = prob;
else
	[f,g] = g5_true(double(x));
end
return


function [f,g] = g5_true(x)
f = 3*x(1) + 0.000001*x(1)^3 + 2*x(2) + 0.000002/3*x(2)^3;

g(1) = x(4) - x(3) + 0.55;
g(2) = x(3) - x(4) + 0.55;

tmp1 = 1000*sin(-x(3)-0.25) + 1000*sin(-x(4)-0.25) + 894.8 - x(1);
tmp2 = 1000*sin(x(3)-0.25) + 1000*sin(x(3)-x(4)-0.25) + 894.8 - x(2);
tmp3 = 1000*sin(x(4)-0.25) + 1000*sin(x(4)-x(3)-0.25) + 1294.8;

eps = 1.e-1;

g(3) = tmp1 + eps;
g(4) = eps - tmp1;
g(5) = tmp2 + eps;
g(6) = eps - tmp2;
g(7) = tmp3 + eps;
g(8) = eps - tmp2;
return
