
function flag = stability_check(trainx, trainy, trainc, krg_obj, krg_con)
flag = true;
n = length(krg_obj);
for i = 1:n
    [y, sig] = predictor(trainx, krg_obj{i});
     y =  denormzscore(trainy, y);
     deltaf = abs(y - trainy);
     
     if max(deltaf) >0.01
         flag = false;
         return
     end    
end

if ~isempty(krg_con)
    nc = length(krg_con);
    for i = 1:nc
        [c, sig] = predictor(trainx, krg_con{i});       
        deltac = abs(c - trainc);
        
        if max(deltac) > 0.01
            flag = false;
            return
        end
    end
end

end
