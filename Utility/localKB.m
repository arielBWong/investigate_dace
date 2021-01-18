function localKB(prim, arcKB_x, arcKB_y, arcEI_x, arcEI_y, prob)
% utility: use the local bump to form local region
% (1) select points in the first bump
% (2) sample one more point in local region to build surrogate
% (3) use k nearest neighbour to build local surrogate

%----------------

nboxes = length(prim.boxes);
bump_lb = prob.xl_bl;
bump_ub = prob.xl_bu;

% only consider the first region
mentioned_var = prim.boxes{1}.vars;
nb = length(mentioned_var);


if ~isempty(mentioned_var)
    for jj = 1:nb % varible indicator
        if ~isnan(prim.boxes{ii}.min(jj))
            bump_lb(mentioned_var(jj)) =  prim.boxes{ii}.min(jj);
        end
        
        if ~isnan(prim.boxes{ii}.max(jj))
            bump_ub(mentioned_var(jj)) =  prim.boxes{ii}.max(jj);
        end        
    end
end

% find x within bump boundary
collect_x = [arcKB_x; arcEI_x];






end