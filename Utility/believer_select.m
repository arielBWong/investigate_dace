%-----------------------------------------------------
%-- select one point --
function [newx, growflag] = believer_select(popx, trainx, prob, upper, varargin)
% select one point from believer's last population
% and avoid repeated point
% varagin: believer_selected used by hybrid or believer

repeatindex = ismember(popx, trainx, 'row');
n_new = sum(~repeatindex);
if n_new == 0
    fprintf('Evolutional process unable to find new  matching xl');
    growflag = false;
    
    if ~empty(varargin) % believer strategy, no unseen data, no add
        newx = [];
        return;
    end
    [newx, growflag] = newpoint_randselection(prob, trainx, upper);
    return;
end
newindex = ~repeatindex;
new_xl = popx(newindex, :);

newx = new_xl(1,:);
growflag = true;

end