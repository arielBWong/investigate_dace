function [f,g] = g1(x)
if nargin == 0
	prob.nx = 13;
	prob.nf = 1;
	prob.ng = 9;
	prob.range = cell(1, prob.nx);
	for i = 1:9
		prob.range{i} = Range('range', [0,1]);
	end
	prob.range{10} = Range('range', [0,100]);
	prob.range{11} = Range('range', [0,100]);
	prob.range{12} = Range('range', [0,100]);
	prob.range{13} = Range('range', [0,1]);
	f = prob;
else
	[f,g] = g1_true(double(x));
end
return


function [f,g] = g1_true(x)
f = 5*sum(x(1:4)) - 5*(x(1:4)*x(1:4)') - sum(x(5:13));
g(1) = 10 - (2*x(1) + 2*x(2) + x(10) + x(11));
g(2) = 10 - (2*x(1) + 2*x(3) + x(10) + x(12));
g(3) = 10 - (2*x(2) + 2*x(3) + x(11) + x(12));
g(4) = 8*x(1) - x(10);
g(5) = 8*x(2) - x(11);
g(6) = 8*x(3) - x(12);
g(7) = 2*x(4) + x(5) - x(10);
g(8) = 2*x(6) + x(7) - x(11);
g(9) = 2*x(8) + x(9) - x(12);
return 
