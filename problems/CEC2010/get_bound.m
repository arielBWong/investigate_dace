function[lower, upper] = get_bound(dim, cfunc_num)
switch cfunc_num
   
    case 1
        lower = zeros(1, dim);
        upper = ones(1, dim) * 10;
        
    case 2
        lower = ones(1, dim) * -5.12;
        upper = ones(1, dim) * 5.12;
        
    case 3
        lower = ones(1, dim) * -1000;
        upper = ones(1, dim) * 1000;
        
    case 4
        lower = ones(1, dim) * -50;
        upper = ones(1, dim) * 50;
        
    case 5
        lower = ones(1, dim) * -600;
        upper = ones(1, dim) * 600;
        
    case 6
        lower = ones(1, dim) * -600;
        upper = ones(1, dim) * 600;
    
    case 7
        lower = ones(1, dim) * -140;
        upper = ones(1, dim) * 140;
        
    case 8
        lower = ones(1, dim) * -140;
        upper = ones(1, dim) * 140;      
        
    case 9
        lower = ones(1, dim) * -500;
        upper = ones(1, dim) * 500;
        
    case 10
        lower = ones(1, dim) * -500;
        upper = ones(1, dim) * 500;
        
    case 11
        lower = ones(1, dim) * -100;
        upper = ones(1, dim) * 100;
        
    case 12
        lower = ones(1, dim) * -1000;
        upper = ones(1, dim) * 1000;
        
    case 13
        lower = ones(1, dim) * -500;
        upper = ones(1, dim) * 500;
        
   case 14
        lower = ones(1, dim) * -1000;
        upper = ones(1, dim) * 1000;
        
   case 15
        lower = ones(1, dim) * -1000;
        upper = ones(1, dim) * 1000;
    
    case 16
        lower = ones(1, dim) * -10;
        upper = ones(1, dim) * 10;
        
   case 17
        lower = ones(1, dim) * -10;
        upper = ones(1, dim) * 10;
        
   case 18
        lower = ones(1, dim) * -50;
        upper = ones(1, dim) * 50;
        
               
        
        
end

end