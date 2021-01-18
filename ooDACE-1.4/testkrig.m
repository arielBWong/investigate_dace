function testkrig
N = 10;

LB = 8.0;UB = 9.0;

X = (linspace(LB,UB,N))';
X=sort(X);
Y = feval(@testfunc,X);

krige=oodacefit(X,Y);
Xpred = (linspace(LB,UB,1000))';
Xpred=sort(Xpred);

[Ypred, ~,~,~] = predictor(Xpred, krige);

figure;hold on;
plot(X,Y,'r.-');
plot(Xpred,Ypred,'b.-');
legend('Original','Predicted');

return

function y = testfunc(x)
y = [-216 [1 1 1 1 1 1 1 1 1]*0]';
% y=sum(x.^2,2)+10*sin(x);

return