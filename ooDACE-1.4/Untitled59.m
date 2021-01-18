load('data')
x = data(:, 1);
y = data(:, 2);

krg = oodacefit( x, y);




xt = linspace(-5.12, 5.12, 100);
yt = dace_predict(xt', krg);
%yt = denormzscore(y, yt);

plot(xt', yt, 'r--'); hold on;
plot(x, y, 'bo');