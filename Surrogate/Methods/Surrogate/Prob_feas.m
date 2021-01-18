% Computes the probability of feasibility
function [P,p] = Prob_feas(g_pred,g_sig)
% g_pred <=0 is feasible
p = zeros(1,size(g_pred,2));
for i=1:size(g_pred,1)
    for j=1:size(g_pred,2)
        if g_sig(i,j)~=0
            z = (0-g_pred(i,j))./g_sig(i,j);
            p(i,j) = 1-0.5 * erfc(z./sqrt(2));
        elseif g_pred(i,j)<=0
            p(i,j) = 1;
        end
    end
end
P = sum(p,2)/size(g_pred,2);
return
