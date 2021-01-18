% ---demo test: forrestor
function[] = processplot1d(fighn, trainx, trainy, krg, prob, initx, before, daceflag)

crosscheck(krg{1}, trainx, trainy, daceflag);
clf(fighn);
% (1) create test
testdata = linspace(prob.xl_bl, prob.xl_bu, 100);
testdata = testdata';

% (2) predict
[fpred, sig] = surrogate_predict(testdata, krg, daceflag);


yyaxis left;
fpred = denormzscore(trainy, fpred);
plot(testdata, fpred, 'r--', 'LineWidth', 1); hold on;

% sig variation
y1 = fpred + sig;
y2 = fpred - sig;
y = [y1', fliplr(y2')];
x = [testdata', fliplr(testdata')];
fill(x, y, 'r', 'FaceAlpha', 0.1, 'EdgeColor','none'); hold on;

% (3) real
[freal, sig]= prob.evaluate_l([], testdata);
plot(testdata, freal, 'b', 'LineWidth', 2);hold on;

% (4) scatter train
scatter(trainx, trainy, 80, 'ko', 'LineWidth', 2);

inity = prob.evaluate_l([], initx);
scatter(initx, inity, 40, 'ro', 'filled');
%---
newy = prob.evaluate_l([], before);
scatter(before, newy, 80, 'ko', 'filled');

% (5) calculate EI and plot
yyaxis right;
[train_ynorm, ~, ~] = zscore(trainy);

ynorm_min = min(train_ynorm);
fprintf('rebuild landscape best f: %f\n',ynorm_min );
if daceflag
    fit = EIM_evaldace(testdata, ynorm_min,  krg, []);
else
    fit = EIM_eval(testdata, ynorm_min,  krg, []);
end
fit = -fit;
plot(testdata, fit, 'g--');

data = [trainx, trainy];
save('data', 'data');
pause(0.5);
end

function crosscheck(krg, trainx, trainy, daceflag)
if daceflag
    [yn, ~] = predictor(trainx, krg);
else
    [yn, ~] = predict(krg, trainx);
end

y =  denormzscore(trainy, yn);
b = max(abs(y - trainy)) - 0.01;
disp(b);
fprintf('stability check should be negative: %f\n',b);
% save('sig_dace', 'sig_dace');

a = unique(round(trainx, 3));
% size(a)
% size(trainx)
end