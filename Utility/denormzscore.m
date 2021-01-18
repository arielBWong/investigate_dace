
 function f = denormzscore(trainy, fnorm)
[train_y_norm, y_mean, y_std] = zscore(trainy);
f = fnorm * y_std + y_mean;
 end