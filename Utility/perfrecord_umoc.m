function  perfrecord_umoc(xu, xl, fu, fc, prob, seed, method, nxu, nxl, varargin)
% save nd front
%

% nxu: number of function evaluation upper
% nxl: number of function evaluation lower


% compatible for saving lower level
if strcmp(method(1:2), 'll') % for paper deceptive analysis
    lowersave(xl, fu, fc, prob, seed, method,varargin{1});
    return 
end


num = length(prob.xl_bl);
savepath = strcat(pwd, '\result_folder\', prob.name, '_',method);
n = exist(savepath);
if n ~= 7
    mkdir(savepath)
end


% save original results
savename_xu = strcat(savepath, '\xu_raw_', num2str(seed),'.csv');
csvwrite(savename_xu, xu);
savename_xl = strcat(savepath, '\xl_raw_', num2str(seed),'.csv');
csvwrite(savename_xl, xl);
savename_fu = strcat(savepath, '\fu_raw_', num2str(seed),'.csv');
csvwrite(savename_fu, fu);
savename_fc = strcat(savepath, '\fc_raw_', num2str(seed),'.csv');
csvwrite(savename_fc, fc);


% extract nd front
num_con = size(fc, 2);
if ~isempty(fc) % constraint problems
    index_c = sum(fc <= 0, 2) == num_con;
    if sum(index_c) ~=0  % exist feasible,
        feasible_y = fu(index_c, :);
        feasible_x = xu(index_c, :);
        feasible_c =fc(index_c, :);
        
        nd_index = Paretoset(feasible_y);
        fu_nd = feasible_y(nd_index, :);
        xu_nd = feasible_x(nd_index, :);
        fc_nd = feasible_c(nd_index, :);
        
    else
        fu_nd =[];
        xu_nd = [];
        fc_nd = [];
    end
else % non constraint upper problem/ nd always exists
    nd_index = Paretoset(fu);
    fu_nd = fu(nd_index, :);
    xu_nd = xu(nd_index, :);
    fc_nd = [];
end

nn = [nxu, nxl];
% save nd front
savename_xu = strcat(savepath, '\xu_', num2str(seed),'.csv');
savename_fu = strcat(savepath, '\fu_', num2str(seed),'.csv');
savename_fc = strcat(savepath, '\fc_', num2str(seed),'.csv');
savename_nn = strcat(savepath, '\nn_', num2str(seed),'.csv');
csvwrite(savename_xu, xu_nd);
csvwrite(savename_fu, fu_nd);
csvwrite(savename_fc, fc_nd);
csvwrite(savename_nn, nn);
end