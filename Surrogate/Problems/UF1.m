 function [f,g] = UF1(x)
if nargin ==0
	prob.nf = 2;
	prob.ng = 0;
	prob.nx = 3;
    for i = 1:prob.nx
        if i == 1
            prob.bounds(i,:) = [0,1];
        else            
            prob.bounds(i,:) = [-1,1];
        end
    end

	f = prob;
    g = [];
else
	[f,g] = UF1_true(x);
end
return

function [f,g] = UF1_true(x)
D = size(x,2);
J1 = 3 : 2 : D;
J2 = 2 : 2 : D;
Y  = x - sin(6*pi*repmat(x(:,1),1,D)+repmat(1:D,size(x,1),1)*pi/D);
f(:,1) = x(:,1)         + 2*mean(Y(:,J1).^2,2);
f(:,2) = 1-sqrt(x(:,1)) + 2*mean(Y(:,J2).^2,2);
g = [];
return
