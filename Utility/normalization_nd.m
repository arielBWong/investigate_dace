function train_y_norm = normalization_nd(train_y)
num_y = size(train_y, 1);
index_p = Paretoset(train_y);
nd_front =  train_y(index_p, :);
if size(nd_front, 1)>1
    ideal = min(nd_front);
    nadir = max(nd_front);
    train_y_norm = (train_y - ideal) ./ (nadir - ideal);
else
    train_y_norm = normalization_y(train_y);
end
end