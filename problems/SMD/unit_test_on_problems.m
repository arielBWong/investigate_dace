%%
% problem testing
% to compare with python implementation
% usage: change prob to smd1-12
% copy generated trainx.csv to python project
% compare obj and con
% the readcsv is to eliminate decimal difference

prob = smd6();

init_size = 30;
rng(seed, 'twister');
[f,c] = prob.evaluate_u([0,0],[-5, -5, 0]);

% xu = [0, 0];
% l_nvar = prob.n_lvar;
% upper_bound = prob.xl_bu;
% lower_bound = prob.xl_bl;
% xu_init = repmat(xu, init_size, 1);
% train_xl = lhsdesign(init_size,l_nvar,'criterion','maximin','iterations',1000);
% train_xl = repmat(lower_bound, init_size, 1) + repmat((upper_bound - lower_bound), init_size, 1) .* train_xl;
% 
% % test only---
% testname = 'trainx.csv';
% csvwrite(testname,train_xl);
% % test only-----
% 
% % evaluate training fl from xu_init and train_xl
% % compatible with non-constriant problem
% train_xl = csvread(testname);
% [train_fl, train_fc] = prob.evaluate_l(xu_init, train_xl);
% pause=0;


%-------------------------------------------------------------------------- 
xl = [0, 0, 1];
u_nvar = prob.n_uvar;
upper_bound = prob.xu_bu;
lower_bound = prob.xu_bl;
xl_init = repmat(xl, init_size, 1);

train_xu = lhsdesign(init_size,u_nvar,'criterion','maximin','iterations',1000);
train_xu = repmat(lower_bound, init_size, 1) + repmat((upper_bound - lower_bound), init_size, 1) .* train_xu;
% test only---
testname = 'trainx.csv';
csvwrite(testname,train_xu);
% test only-----
train_xu = csvread(testname);
[train_fu, train_fc] = prob.evaluate_u(train_xu, xl_init);


pause=0;
