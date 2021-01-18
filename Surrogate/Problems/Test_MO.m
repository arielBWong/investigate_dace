% Test problem is defined here
function [f,g] = Test_MO(x)
if nargin == 0
    prob.nf = 2;
    prob.ng = 0;
    prob.nx=2;
    for i = 1:prob.nx
        prob.bounds(i,:) = [0,4];
    end
    f = prob;
    g = [];
else
    [f,g] = Test_MO_true(x);
end
return

function [f,g] = Test_MO_true(x)
f(:,1)=x(:,1).^2+x(:,2);
f(:,2)=(x(:,1)-2).^2+x(:,2);
g = [];
return