% Problem
param.prob_name='Welded_beam_MO';
param.strategy=1; % 1 is Kriging Believer ; 2 is Expected Improvement ; 3 Hybrid

% param.GPR_type=1 for GPR of Matlab; 2 for DACE
param.GPR_type=2;

% In generating the GPR model, no of mulistart attempts
param.no_trials=1;

% This is to be switched on only when support in required with Excel I/O:
% Human in Loop
param.HIL=1;

% These are optimization parameters
param.popsize=50;param.MFE=55; 
param.seed=11;

% These are subEA parameters
param.EA_popsize=100;param.EA_generations=20; 
param.dist_threshold=0.01;

% These are the parameters for DE and PM used for offspring generation
param.DE_crossover_rate=0.9;
param.DE_mutation_factor=0.5;
param.Poly_mutation_prob=1; % Setting it one uses 1/D
param.Poly_mutation_distindex=30;

% Local search flag
param.local_search=1; % Set 0 if you want to skip

% SQP Parameters for surrogate assisted local search if local search flag is 1
param.SQP_Maxiter=1000;


