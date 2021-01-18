function save_eachmodel(seed, method, mdl_save, init_size, prob,varargin)

num = length(prob.xl_bl);
savepath = strcat(pwd, '\resultfolder_gp\', prob.name, '_', num2str(num) ,'_',method);
savepath = strcat(savepath, '_init_', num2str(init_size), '_modelsave');

n = exist(savepath);
if n ~= 7
    mkdir(savepath)
end


savemodel_eachseed = strcat(savepath, '\seed_', num2str(seed));
n = exist(savemodel_eachseed);
if n ~= 7
    mkdir(savemodel_eachseed)
end

iter_size = length(mdl_save);

for i = 1 : iter_size
mdl = mdl_save{i};
savename = strcat(savemodel_eachseed, '\mdl_', num2str(i) );
save(savename, 'mdl');
end

% record swing between KB and EI
if ~isempty(varargin)
    switchrecord = varargin{1};
    switchrecord = switchrecord';
    savename = strcat(savepath, '\switch_seed_', num2str(seed), '.csv');
    csvwrite(savename, switchrecord);
end




end