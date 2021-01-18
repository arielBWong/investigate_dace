% This function finds best feasible or objectives of ND solutions
function [f_ref]=Find_reference(archive,prob)
X=archive.x;F=archive.muf;id1=[];
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
    if(prob.nf==1)
        [~,id1]=sort(F(feas_ids));
    else
        [id_fronts,~,~,~] = E_NDSort_c(F(feas_ids,:));
        id1=cell2mat(id_fronts);
        id1=feas_ids(id1);
    end
end
if(~isempty(id1))
    if(prob.nf==1)
        f_ref=archive.muf(id1(1));
    else
        f_ref=archive.muf(feas_ids(id_fronts{1}),:);
    end
else
    f_ref=[];
end
return

