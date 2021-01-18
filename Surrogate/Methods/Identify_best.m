function x=Identify_best(archive,s_pop,param,prob)
if(prob.nf==1) && (param.strategy==1)
    % The best solution is the first solution of the population
    new.x=s_pop.x(1,:);
end

if(prob.nf>1) && (param.strategy==1)
    % The best solution is the one which gives max HV contribution
    X=archive.x;
    F=archive.muf;
    if(prob.ng>0)
        G=[archive.mug];
        CV=sum(G.*(G>=0),2);
        infeas_ids=[find(CV>0)]';
        [~,id2]=sort(CV(infeas_ids));
        id2=infeas_ids(id2);
        feas_ids=setdiff([1:size(archive.x,1)],infeas_ids);
    else
        feas_ids=1:size(archive.x,1);
    end
    
    if(~isempty(feas_ids))
        ref_point=1.1*ones(1,prob.nf);
        [id_fronts,~,~,~] = E_NDSort_c(F(feas_ids,:));
        nd_ids=feas_ids(id_fronts{1});
        ideal=min(F(nd_ids,:));nadir=max(F(nd_ids,:));
        nftrue_nd=(F(nd_ids,:)-repmat(ideal,length(nd_ids),1))./(repmat(nadir-ideal,length(nd_ids),1));
        nfpopf=(s_pop.muf-repmat(ideal,size(s_pop.muf,1),1))./(nadir-ideal);
        performance=compute_hv(nftrue_nd,nfpopf,ref_point);
        
        % Identifying the best performing solution
        [~,id]=max(performance);
        new.x=s_pop.x(id,:);
        [new.muf,tmp]=feval(param.prob_name,new.x);
        [new.sf]=zeros(1,prob.nf);
        if(prob.ng>0)
            new.mug=tmp;
            new.sg=zeros(1,prob.ng);
        end
    else
        new.x=s_pop.x(1,:);
        [new.muf,tmp]=feval(param.prob_name,new.x);
        [new.sf]=zeros(1,prob.nf);
        if(prob.ng>0)
            new.mug=tmp;
            [new.sg]=zeros(1,prob.ng);
        end
    end
end

if(param.strategy==2)
    % The best solution is the first solution of the population
    new.x=s_pop.x(1,:);
end
x=new.x;

return
