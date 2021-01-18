function mdl = surrogate_train(x, y, daceflag)

num_mdl = size(y, 2);
mdl = cell(1, num_mdl);
for ii = 1:num_mdl
    if daceflag
        mdl{ii} = oodacefit(x,y(:,ii));
    else
        mdl{ii} = myfitgrp(x,y(:,ii),5);
    end
end
end