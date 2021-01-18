%-----auxiliary function ---
function [f, s] = surrogate_predict(x, mdl, daceflag)
% frame for predicting mean from surrogate mdl

num_mdl = length(mdl);   
num_x = size(x, 1);
f = zeros(num_x, num_mdl);
s = zeros(num_x, num_mdl);
for ii =1:num_mdl
    if daceflag %dace
        if size(x, 1) > 1
            [f(:, ii), s(:, ii)] = predictor(x, mdl{ii});
        else
            [f(:, ii), ~ , s(:, ii)] =  predictor(x, mdl{ii});
        end
    else % gpr       
        [f(:, ii), s(:, ii)] =  predict(mdl{ii}, x);
        
    end
end
end

