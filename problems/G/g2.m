function [f,g] = g2(x)
if nargin == 0
	prob.nx = 20;
	prob.nf = 1;
	prob.ng = 2;
	prob.range = cell(1, prob.nx);
	for i = 1:prob.nx
		prob.range{i} = Range('range', [0,10]);
	end
	f = prob;
else
	[f,g] = g2_true(double(x));
end
return


function [f,g] = g2_true(x)
tmp1 = 0;
tmp2 = 1;
tmp3 = 0;
for i = 1:length(x)
	tmp1 = tmp1 + cos(x(i))^4;
	tmp2 = tmp2 * cos(x(i))^2;
	tmp3 = tmp3 + i*x(i)^2;
end
f = -abs((tmp1 - 2*tmp2) / sqrt(tmp3));
g(1) = prod(x) - 0.75;
g(2) = 7.5*length(x) - sum(x);
return
