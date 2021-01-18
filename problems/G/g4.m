%% Himmelblau function
function [f,g] = g4(x)
if nargin == 0
	prob.nx = 5;
	prob.nf = 1;
	prob.ng = 6;
	prob.range = cell(1, prob.nx);
	prob.range{1} = Range('range', [78,102]);
	prob.range{2} = Range('range', [33,45]);
	prob.range{3} = Range('range', [27,45]);
	prob.range{4} = Range('range', [27,45]);
	prob.range{5} = Range('range', [27,45]);
	f = prob;
else
	[f,g] = g4_true(double(x));
end
return


function [f,g] = g4_true(x)
f = 5.3578547*x(3)^2 + 0.8356891*x(1)*x(5) + 37.293239*x(1) - 40792.141;
tmp1 = 85.334407 + 0.0056858*x(2)*x(5) + 0.0006262*x(1)*x(4) - 0.0022053*x(3)*x(5);
tmp2 = 80.51249 + 0.0071317*x(2)*x(5) + 0.0029955*x(1)*x(2) + 0.0021813*x(3)^2;
tmp3 = 9.300961 + 0.0047026*x(3)*x(5) + 0.0012547*x(1)*x(3) + 0.0019085*x(3)*x(4);
g(1) = tmp1;
g(2) = 92 - tmp1;
g(3) = tmp2 - 90;
g(4) = 110 - tmp2;
g(5) = tmp3 - 20;
g(6) = 25 - tmp3;
return
