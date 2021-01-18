function [f,g] = UF5_a(x)
if nargin == 0
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
	[f,g] = UF5_a_true(x);
end
return

function [f,g] = UF5_a_true(x)
D = size(x,2); 
J1 = 3 : 2 : D;
J2 = 2 : 2 : D;
Y  = x - sin(pi*repmat(x(:,1),1,D)+repmat(1:D,size(x,1),1)*pi/D);
hY = 2*Y.^2 - cos(pi*Y) + 1;
f(:,1) = x(:,1)   + (1/2+0.1)*abs(sin(pi*x(:,1)))+2*mean(hY(:,J1),2);
f(:,2) = 1-x(:,1) + (1/2+0.1)*abs(sin(pi*x(:,1)))+2*mean(hY(:,J2),2);
g = [];
return