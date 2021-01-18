classdef mobp7
    properties
        p = 1;
        q = 2;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'mobp7';
        uopt = NaN;
        lopt = 0; % double check needed
    end
    methods
        function obj = mobp7()
            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p) * 10.0 ;
           
           
            % init bound lower level
            obj.xl_bl = zeros(1, obj.q);
            obj.xl_bu = ones(1, obj.q) * 40;      
        end
        
        function [f, c] = evaluate_u(obj, xu, xl) 
            %-obj
            f(:, 1) = xl(:, 1) + xl(:,  2).^2 + xu + (sin(xl(:, 1) + xu)).^2 ;
            f(:, 2) = cos(xl(:, 2)) .*(0.1 + xu) .* exp(-xl(:, 1) ./ (0.1 + xl(:, 2)));
                 
            %-cie
            c= [];            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)          
            %-obj
            f1 = ( (xl(:, 1)-2).^2 + (xl(:,2)-1).^2 )/4;
            f2 = ( xl(:, 2).* xu + (5-xu).^2 )/16;
            f3 = sin(xl(:, 2)/10.0);
            f4 = ( xl(:, 1).^2 + (xl(:, 2) - 6).^4 - 2*xl(:, 1).*xu - (5-xu).^2 )/80;
            f = f1 + f2 + f3 + f4;
            
            %-cie
           c(:, 1) = xl(:, 1).^2 - xl(:, 2);
           c(:, 2) = 5*xl(:, 1) .^ 2 + xl(:, 2) - 10;
           c(:, 3) = xl(:, 2) + xu/6- 5;
        
         end
    end
end
