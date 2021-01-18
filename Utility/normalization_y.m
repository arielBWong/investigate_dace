function train_y_norm = normalization_y(train_y_norm)
num_y = size(train_y_norm, 1);
train_y_norm = (train_y_norm -repmat(min(train_y_norm),num_y,1))./...
    repmat(max(train_y_norm)-min(train_y_norm),num_y,1);
end