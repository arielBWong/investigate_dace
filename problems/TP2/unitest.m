%% problem test
prob = tp3(1,1);
lb1 = prob.xl_bl(1);
ub1 = prob.xl_bu(1);
xl =linspace(lb1, ub1, 101);

f = prob.evaluate_l([], xl');
plot(xl, f); 

xu = [1,1 ];
fm = prob.evaluate_l([], [0.5, 0.5])
rng(1, 'twister');
[match_xl, n_fev, flag] = llmatch(xu, prob, 20, 20, 'EIMnext', 60, 'EIM_eval', 1)
rng(1, 'twister');
[match_xl, n_fev, flag] = llmatch_sao_archiveinsert(xu, prob, 20, 20, 60, 1)

 a = 0;