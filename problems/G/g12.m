function [f,g] = g12(x)
if nargin == 0
	prob.nx = 3;
	prob.nf = 1;
	prob.ng = 1;
	prob.range = cell(1, prob.nx);
	for i = 1:prob.nx
		prob.range{i} = Range('range', [0,10]);
	end
	f = prob;
else
	[f,g] = g12_true(double(x));
end
return


function [f,g] = g12_true(x)
sum=0;
flag=0;
f = -(100 - (x(1)-5)^2 - (x(2)-5)^2 - (x(3)-5)^2) / 100;
for i = 1:9
	for j = 1:9
		for k = 1:9
			gg = 0.0625 - ((x(1)-i)^2 + (x(2)-j)^2 + (x(3)-k)^2);
            if(gg>=0)
                flag=1;
                g=0;
                break;
            end
            sum=sum+gg*-1;
        end
%         if flag ==1 
%             break
%         end
    end
%         if flag ==1 
%             break
%         end
end
if (flag==1)
    g=0;
else
    g=sum;
end
return
