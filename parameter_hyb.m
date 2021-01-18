param_hyb = struct();
param_hyb.num_pop = 100;
param_hyb.num_gen = 100;
param_hyb.init_size = 32;
param_hyb.iter_size = 140; %140
param_hyb.norm_str = 'normalization_z';
param_hyb.par = 0.01;
save('param_hyb', 'param_hyb');