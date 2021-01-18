function Test_Approximator

clear all;close all;clc;
addpath Problems
addpath Methods
addpath 'Methods\Surrogate'

load('data.mat');
prob.ng=0;prob.nf=1;
pop.x=data(:,1);
pop.muf=data(:,2);
pop.mug=[];


% param.prob_name='Test_Forrester';
% param.strategy=1; % 1 is Kriging Believer ; 2 is Expected Improvement ; 3 Hybrid
% 
% % param.GPR_type=1 for GPR of Matlab; 2 for DACE
    param.GPR_type=2;
% 
% % No of Multistarts to optimize the hyperparameters in the GPR model
    param.no_trials=5;
% 
% % Setting the random seed
% param.seed=10;rng(param.seed);
% 
% % Problem parameters
% prob=feval(param.prob_name);
% 
% % Number of datapoints
% param.popsize=10;
% 
% % Initialize population and evaluate it
% pop=Initialize_pop(prob,param.popsize);pop.mug=[];
% [pop.muf,pop.mug]=feval(param.prob_name,pop.x);
% pop.sf=zeros(size(pop.x,1),prob.nf);
% pop.sg=zeros(size(pop.x,1),prob.ng);

% Passing it over to archive
archive.x=pop.x;
archive.muf=pop.muf;
archive.mug=pop.mug;

% Training the GPR Model
gprMdl=Train_GPR(archive.x,[archive.muf archive.mug],param);


plot(pop.x,pop.muf,'mo'); hold on;
child.x=linspace(-5.12,5.12,1000);child.x=child.x';
true_f=test_Rastri(child.x);
plot(child.x,true_f,'b-'); hold on;


%child.x=archive.x;

% Predicting the performance of the GPR Model
% child=Initialize_pop(prob,param.popsize);
[mu,sigma]=Predict_GPR(gprMdl,child.x,param,archive);
% [mu,sigma]=Predict_GPR(gprMdl,pop.x,param,archive);
%  plot(pop.x, mu,'r.' );
child.muf=mu(:,1:prob.nf);
child.sf=sigma(:,1:prob.nf);
if(prob.ng>0)
    child.mug=mu(:,prob.nf+1:end);
    child.sg=sigma(:,prob.nf+1:end);
end


plot(child.x,child.muf,'r.'); 
return

