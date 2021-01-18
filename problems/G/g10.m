function [f,g] = g10(x)
if nargin == 0
	prob.nx = 8;
	prob.nf = 1;
	prob.ng = 6;
	prob.range = cell(1, prob.nx);
	prob.range{1} = Range('range', [100,10000]);
	prob.range{2} = Range('range', [1000,10000]);
	prob.range{3} = Range('range', [1000,10000]);
	prob.range{4} = Range('range', [10,1000]);
	prob.range{5} = Range('range', [10,1000]);
	prob.range{6} = Range('range', [10,1000]);
	prob.range{7} = Range('range', [10,1000]);
	prob.range{8} = Range('range', [10,1000]);
	f = prob;
else
	[f,g] = g10_true(double(x));
end
return


function [f,g] = g10_true(x)
f = x(1)+x(2)+x(3);
g(1) = 1 - 0.0025*(x(4)+x(6));
g(2) = 1 - 0.0025*(x(5)+x(7)-x(4));
g(3) = 1 - 0.01*(x(8)-x(5));
g(4) = x(1)*x(6) - 833.33252*x(4) - 100*x(1) + 83333.333;
g(5) = x(2)*x(7) - 1250*x(5) - x(2)*x(4) + 1250*x(4);
g(6) = x(3)*x(8) - 1250000 - x(3)*x(5) + 2500*x(5);
return
