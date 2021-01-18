function flag=Check_distance(x,BLx,prob,param)
nBLx=(BLx-prob.bounds(:,1)')./(prob.bounds(:,2)'-prob.bounds(:,1)');
nx=(x-repmat(prob.bounds(:,1)',size(x,1),1))./(repmat(prob.bounds(:,2)'-prob.bounds(:,1)',size(x,1),1));
flag=1;
distances=sqrt(sum((repmat(nBLx,size(x,1),1)-nx).*(repmat(nBLx,size(x,1),1)-nx),2));
% If the minimum distance in normalized variable space is less <=
% sqrt(nobj) choose EI sampling point
if(min(distances)<=param.dist_threshold)
    flag=2;
end
return
