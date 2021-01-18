function train_y_norm = normalization_z(train_y)
% single normalization with zscore
if size(train_y, 2)> 1
    error('zscore is not for mo');
end
[train_y_norm,~, ~] = zscore(train_y, 0, 1); 
end