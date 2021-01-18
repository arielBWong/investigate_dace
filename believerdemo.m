%% 
 % prob = Forrestor();
 clc;
 clear all;

% prob = 'levy(2,2)';
prob = 'Shekel(1, 1)';

% prob = 'ackley(1, 1)';
% prob = 'tp3(2,2)';
% prob = 'rastrigin(1,1)';
% prob = 'Zakharov(2,2)';
% prob = 'Shekel_curve()';
% [match_xl, n_fev, flag]   = ego_EIdace([0, 0], prob, 2);
[match_xl, n_fev, flag]    = ego_EIdaceV11([0, 0], prob, 3);



