% Test problem is defined here
function [f,g] = Test_Forrester(x)
if nargin == 0
    prob.nf = 1;
    prob.ng = 0;
    prob.nx=1;
    for i = 1:prob.nx
        prob.bounds(i,:) = [0,1];
    end
    f = prob;
    g = [];
else
    [f,g] = Test_Forrester_true(x);
end
return

function [f,g] = Test_Forrester_true(x)
f(:,1)=((6*x(:,1)-2).^2).*sin(12*x(:,1)-4);
g = [];
return




