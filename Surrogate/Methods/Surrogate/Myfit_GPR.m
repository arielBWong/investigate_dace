% mygpr function that optimizes the model parameters using multistart fmincon
function gprMdl=Myfit_GPR(xx,yy,no_trials)
f = @(x)gprobj_function(x,xx,yy);warning('off');
options = optimoptions(@fmincon,'Display','off');
LB=1e-30*ones(1,size(xx,2)+1);
UB=3*ones(1,size(xx,2)+1);
best_f=1e20;
for j=1:no_trials
    xstart=LB+(UB-LB).*rand(1,length(LB));
    [xn,fn] = fmincon(f,xstart,[],[],[],[],LB,UB,[],options);
    if(fn<best_f)
        best_f=fn;
        new_x=xn;
    end
end
% Trying the recommended value: Initial sigmas are the length scale parameters: Recommended
xstart=[std(xx) std(yy)/sqrt(2)];
[xn,fn] = fmincon(f,xstart,[],[],[],[],LB,UB,[],options);
if(fn<best_f)
    best_f=fn;
    new_x=xn;
end
% This is the optimal parameters for the model
sigmas=new_x;
eps=1e-30;fac=1;count=1;flag=0;
while flag==0
    try
        flag=1;
        gprMdl=fitrgp(xx,yy,'Standardize',1,'BasisFunction','none','KernelFunction','ardsquaredexponential','FitMethod','none','Sigma',eps*fac,'PredictMethod','exact','KernelParameters',sigmas');
    catch ME
        if (strcmp(ME.identifier,'stats:classreg:learning:impl:GPImpl:GPImpl:UnableToComputeLFactorExact'))
            flag=0;
            fac=10^count;
            count=count+1;
        end
    end
end
return

function [obj_fn]=gprobj_function(sigmas,x,f)
eps=1e-30;fac=1;count=1;flag=0;
while flag==0
    try
        flag=1;
        gprMdl=fitrgp(x,f,'Standardize',1,'BasisFunction','none','KernelFunction','ardsquaredexponential','FitMethod','none','Sigma',eps*fac,'PredictMethod','exact','KernelParameters',sigmas');
    catch ME
        if (strcmp(ME.identifier,'stats:classreg:learning:impl:GPImpl:GPImpl:UnableToComputeLFactorExact'))
            flag=0;
            fac=10^count;
            count=count+1;
        end
    end
end
obj_fn=gprMdl.Impl.computeLogLikelihoodExact();
obj_fn=obj_fn*-1;
return
        