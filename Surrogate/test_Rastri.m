function f=test_Rastri(xl)
 %-obj
 f = 0;

 for i = 1:1
     fb = xl(:, i).^2 - 10 * cos(2 * pi * xl(:, i));
     f = f+ fb ;
 end
 
 f = 10  + f;

return