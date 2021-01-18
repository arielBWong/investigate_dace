% Predicting using a Kriging model
function [y_prd,s_prd]=Predict_GPR(gprMdl,x_tst,param,archive)
y_prd=[];s_prd=[];
if(length(fieldnames(archive)))>3
    response=[archive.muf archive.mug];
else
    response=[archive.muf];
end

for i=1:length(gprMdl)
    if(param.GPR_type==1)
        [tmp_mu,tmp_sigma]=predict(gprMdl{i},x_tst);
    else
        [tmp_mu,tmp_sigma]=Predict_DACE(gprMdl{i},x_tst,i);
    end
    
    % Replacing the predicted values by true values if they have been evaluated
    for j=1:size(x_tst,1)
        tmp=repmat(x_tst(j,:),size(archive.x,1),1)-archive.x;
        d=sum(tmp.*tmp,2);
        id=find(d==0);
        if ~isempty(id)
            tmp_mu(j)=response(id,i);
            tmp_sigma(j)=0;
        end
    end    
    y_prd=[y_prd tmp_mu];
    s_prd=[s_prd tmp_sigma];
end
return
