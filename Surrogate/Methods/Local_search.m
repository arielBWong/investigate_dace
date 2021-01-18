function x=Local_search(x,param,prob,archive,gprMdl)
if (param.strategy==1)
    if(prob.nf==1)
       % Perform a local search on surrogate objective and constraint
       % function
       LB = [prob.bounds(:,1)]';UB = [prob.bounds(:,2)]';
       objfun = @(x)Objective_KB_1(x,param,prob,archive,gprMdl);
       if(prob.ng>0)
           confun = @(x)Constraint_KB_1(x,param,prob,archive,gprMdl);
       end
       options = optimoptions('fmincon','Algorithm','sqp','MaxIterations',param.SQP_Maxiter,'Display','off');
       if(prob.ng>0)
           [x_mod,~] = fmincon(objfun,x,[],[],[],[],LB,UB,confun,options);
       else
           [x_mod,~] = fmincon(objfun,x,[],[],[],[],LB,UB,[],options);
       end
    else
        [f_ref]=Find_reference(archive,prob);
        if(isempty(f_ref))
            % Or since there were no feasible solutions, looking to find
            % feasible
            LB = [prob.bounds(:,1)]';UB = [prob.bounds(:,2)]';
            objfun = @(x)Objective_KB_2E(x,param,prob,archive,gprMdl);
            options = optimoptions('fmincon','Algorithm','sqp','MaxIterations',param.SQP_Maxiter,'Display','off');
            [x_mod,~] = fmincon(objfun,x,[],[],[],[],LB,UB,[],options);
        else
            % Perform a HV contribution maximization subject to feasibile constraints
            LB = [prob.bounds(:,1)]';UB = [prob.bounds(:,2)]';
            objfun = @(x)Objective_KB_2(x,param,prob,archive,gprMdl,f_ref);
            if(prob.ng>0)
                confun = @(x)Constraint_KB_2(x,param,prob,archive,gprMdl);
            end
            options = optimoptions('fmincon','Algorithm','sqp','MaxIterations',param.SQP_Maxiter,'Display','off');
            
            if(prob.ng>0)
                [x_mod,~] = fmincon(objfun,x,[],[],[],[],LB,UB,confun,options);
            else
                [x_mod,~] = fmincon(objfun,x,[],[],[],[],LB,UB,[],options);
            end
        end        
    end
end

if (param.strategy==2)
    if(prob.nf==1)
        % Perform a local search on EI*PF, EI or PF
        LB = [prob.bounds(:,1)]';UB = [prob.bounds(:,2)]';
        objfun = @(x)Objective_EI_1(x,param,prob,archive,gprMdl);
        options = optimoptions('fmincon','Algorithm','sqp','MaxIterations',param.SQP_Maxiter,'Display','off');
        [x_mod,~] = fmincon(objfun,x,[],[],[],[],LB,UB,[],options);
    else
        % Perform a local search on EVVI*PF, EHVI or PF
        LB = [prob.bounds(:,1)]';UB = [prob.bounds(:,2)]';
        objfun = @(x)Objective_EI_2(x,param,prob,archive,gprMdl);
        options = optimoptions('fmincon','Algorithm','sqp','MaxIterations',param.SQP_Maxiter,'Display','off');
        [x_mod,~] = fmincon(objfun,x,[],[],[],[],LB,UB,[],options);
    end
end

% Updating the solution following local search
x=x_mod;
return

function [obj]=Objective_EI_2(x,param,prob,archive,gprMdl)
[f_ref]=Find_reference(archive,prob);
[mu,sigma]=Predict_GPR(gprMdl,x,param,archive);
F=mu(1:prob.nf);
SF=sigma(1:prob.nf);
EI(1)=1;
if(prob.ng>0)
    G=mu(prob.nf+1:end);
    SG=sigma(prob.nf+1:end);
    [PF,~] = Prob_feas(G,SG);
else
    PF=1;
end
% Since  there is already a feasible solution, computing EI
if(~isempty(f_ref))
    f = f_ref;
    [num_pareto,num_obj] = size(f);
    num_x = 1;
    r = 1.1*ones(1, num_obj);
    EI = zeros(num_x,1);
    u=F;mse=SF;
    s = sqrt(max(0,mse));
    r_matrix = repmat(r,num_pareto,1);
    for ii = 1 : num_x
        u_matrix = repmat(u(ii,:),num_pareto,1);
        s_matrix = repmat(s(ii,:),num_pareto,1);
        EIM = (f - u_matrix).*Gaussian_CDF((f - u_matrix)./s_matrix) + s_matrix.*Gaussian_PDF((f - u_matrix)./s_matrix);
        EI(ii) =  min(prod(r_matrix - f + EIM,2) - prod(r_matrix - f,2));
    end    
end
% Maximization of EI or EI*PF
obj=-EI*PF;
return



function [obj]=Objective_KB_2(x,param,prob,archive,gprMdl,f_ref)
[tmp,~]=Predict_GPR(gprMdl,x,param,archive);
F=tmp(1:prob.nf);
if(size(f_ref,1)==1)
    v=f_ref-F;
    v(v<0)=0;
    obj=-1*sum(v);
else
   ideal=min(f_ref);nadir=max(f_ref); 
   ref_point=1.1*ones(1,prob.nf);
   nf_ref=(f_ref-repmat(ideal,size(f_ref,1),1))./(repmat(nadir-ideal,size(f_ref,1),1));
   nf=(F-ideal)./(nadir-ideal);
   performance=compute_hv(nf_ref,nf,ref_point);
   obj=-1*performance;
end

return

function [con,dummy]=Constraint_KB_2(x,param,prob,archive,gprMdl)
dummy=[];
[tmp,~]=Predict_GPR(gprMdl,x,param,archive);  
con=tmp(2:end);
return


function [obj]=Objective_KB_2E(x,param,prob,archive,gprMdl)
[tmp,~]=Predict_GPR(gprMdl,x,param,archive);
tmp(1)=[];
obj=sum(tmp(tmp>0));
return


function [obj]=Objective_EI_1(x,param,prob,archive,gprMdl)
[f_ref]=Find_reference(archive,prob);
[mu,sigma]=Predict_GPR(gprMdl,x,param,archive);
F=mu(1);SF=sigma(1);EI=1;

if(prob.ng>0)
    G=mu(2:end);
    SG=sigma(2:end);
    [PF,~] = Prob_feas(G,SG);
else
    PF=1;
end
if(~isempty(f_ref))
    EI=Expected_Imp(F,SF,f_ref);
end
obj=-EI*PF;
return


function [obj]=Objective_KB_1(x,param,prob,archive,gprMdl)
[tmp,~]=Predict_GPR(gprMdl,x,param,archive);
obj=tmp(1);
return

function [con,dummy]=Constraint_KB_1(x,param,prob,archive,gprMdl)
dummy=[];
[tmp,~]=Predict_GPR(gprMdl,x,param,archive);  
con=tmp(2:end);
return
