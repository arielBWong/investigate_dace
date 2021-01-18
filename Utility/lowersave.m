function  lowersave(x, f, c, prob, seed, method,  varargin)
% save lower match results
% can deal with SO and MO
%---------------------------
num = length(prob.xl_bl);
savepath = strcat(pwd, '\resultfolder_gp\', prob.name, '_', num2str(num) ,'_',method);
% savepath = strcat(pwd, '\resultfolder_gp\', prob.name, '_', num2str(num) ,'_',method);
% savepath = strcat(pwd, '\result_folder\', prob.name, '_',method);
if ~isempty( varargin)
    init_size =  varargin{1};
    savepath = strcat(savepath, '_init_', num2str(init_size));
end


n = exist(savepath);
if n ~= 7
    mkdir(savepath)
end

% check and process for multiple objective
% save nd front only
if size(f, 2)> 1
    % extract nd front
    num_con = size(c, 2);
    if ~isempty(c) % constraint problems
        index_c = sum(c <= 0, 2) == num_con;
        if sum(index_c) ~=0  % exist feasible,
            feasible_y = f(index_c, :);
            feasible_x = x(index_c, :);
            feasible_c = c(index_c, :);
            
            nd_index = Paretoset(feasible_y);
            f_nd = feasible_y(nd_index, :);
            x_nd = feasible_x(nd_index, :);
            c_nd = feasible_c(nd_index, :);
            
        else
            f_nd =[];
            x_nd = [];
            c_nd = [];
        end
    else % non constraint upper problem/ nd always exists
        nd_index = Paretoset(f);
        f_nd = f(nd_index, :);
        x_nd = x(nd_index, :);
        c_nd = [];
    end
    %---------------
    % save nd front
    savename_xu = strcat(savepath, '\xl_', num2str(seed),'.csv');
    savename_fu = strcat(savepath, '\fl_', num2str(seed),'.csv');
    savename_fc = strcat(savepath, '\cl_', num2str(seed),'.csv');

    csvwrite(savename_xu, x_nd);
    csvwrite(savename_fu, f_nd);
    csvwrite(savename_fc, c_nd);
    
    return
end

savename_xu = strcat(savepath, '\xl_', num2str(seed),'.csv');
savename_fu = strcat(savepath, '\fl_', num2str(seed),'.csv');
savename_fc = strcat(savepath, '\cl_', num2str(seed),'.csv');

csvwrite(savename_xu, x);
csvwrite(savename_fu, f);
csvwrite(savename_fc, c);

end