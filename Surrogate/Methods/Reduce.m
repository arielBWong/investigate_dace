% This function identified population for the next generation
function [s_pop]=Reduce(prob,param,c_pop,s_pop,archive)

% This is Kriging Believer
if(param.strategy==1)
    X=[c_pop.x;s_pop.x];
    F=[c_pop.muf;s_pop.muf];
    SF=[c_pop.sf;s_pop.sf];
    id2=[];id1=[];
    if(prob.ng>0)
        G=[c_pop.mug;s_pop.mug];
        SG=[c_pop.sg;s_pop.sg];
        CV=sum(G.*(G>=0),2);
        infeas_ids=[find(CV>0)]';
        [~,id2]=sort(CV(infeas_ids));
        id2=infeas_ids(id2);
        feas_ids=setdiff([1:2*size(s_pop.x,1)],infeas_ids);
    else
        feas_ids=1:2*size(s_pop.x,1);
    end
    if(~isempty(feas_ids))
        if(prob.nf==1)
            [~,id1]=sort(F(feas_ids));
            id1=feas_ids(id1);
        else
            [id_fronts,~,~,~] = E_NDSort_c(F(feas_ids,:));
            id1=cell2mat(id_fronts);
            id1=feas_ids(id1);
        end
    end
    if(~isempty(id1))
        if(size(id1,1)>size(id1,2))
            id1=id1';
        end
    end
    if(~isempty(id2))
        if(size(id2,1)>size(id2,2))
            id2=id2';
        end
    end
    ids=[id1 id2];
    s_pop.x(1:size(s_pop.x,1),:)=X(ids(1:size(s_pop.x,1)),:);
    s_pop.muf(1:size(s_pop.x,1),:)=F(ids(1:size(s_pop.x,1)),:);
    s_pop.sf(1:size(s_pop.x,1),:)=SF(ids(1:size(s_pop.x,1)),:);
    if(prob.ng>0)
        s_pop.mug(1:size(s_pop.x,1),:)=G(ids(1:size(s_pop.x,1)),:);
        s_pop.sg(1:size(s_pop.x,1),:)=SG(ids(1:size(s_pop.x,1)),:);
    end
else
    
    % This is Expected Improvement
    % f_ref is either the best obj of a SO or f_nd for MO
    % f_ref is empty is there are no feasible solutions found so far
    [f_ref]=Find_reference(archive,prob);
    
    X=[c_pop.x;s_pop.x];
    F=[c_pop.muf;s_pop.muf];
    SF=[c_pop.sf;s_pop.sf];
    EI=ones(2*size(s_pop.x,1),1);
    if(prob.ng>0)
        G=[c_pop.mug;s_pop.mug];
        SG=[c_pop.sg;s_pop.sg];
        [PF,~] = Prob_feas(G,SG);
    else
        PF=ones(2*size(s_pop.x,1),1);
    end
    
    % Since  there is already a feasible solution, computing EI
    if(~isempty(f_ref))
        if(prob.nf==1)
            EI=Expected_Imp(F,SF,f_ref);
        else
            f = f_ref;
            [num_pareto,num_obj] = size(f);
            num_x = 2*size(c_pop.x,1);
            r = 1.1*ones(1, num_obj);
            EI = zeros(num_x,1);
            u = zeros(num_x,num_obj);
            mse = zeros(num_x,num_obj);
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
    end
    
    % Maximization of EI or EI*PF
    Merit=EI.*PF;
    [~,ids]=sort(Merit,'descend');
    s_pop.x(1:size(s_pop.x,1),:)=X(ids(1:size(s_pop.x,1)),:);
    s_pop.muf(1:size(s_pop.x,1),:)=F(ids(1:size(s_pop.x,1)),:);
    s_pop.sf(1:size(s_pop.x,1),:)=SF(ids(1:size(s_pop.x,1)),:);
    if(prob.ng>0)
        s_pop.mug(1:size(s_pop.x,1),:)=G(ids(1:size(s_pop.x,1)),:);
        s_pop.sg(1:size(s_pop.x,1),:)=SG(ids(1:size(s_pop.x,1)),:);
    end
end
return








