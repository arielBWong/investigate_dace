% Traning a Kriging model
function gprMdl=Train_GPR(x_trg,response,param)
no_trials=param.no_trials;
for i=1:size(response,2)
    if(param.GPR_type==1)
        gprMdl{i}=Myfit_GPR(x_trg,response(:,i),no_trials);
    else
        gprMdl{i}=Myfit_DACE(x_trg,response(:,i),no_trials);
    end
end
return