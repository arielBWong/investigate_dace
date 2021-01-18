classdef mobp1
    properties
        p = 1;
        q = 2;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'mobp1';
        uopt = NaN;
        lopt = 0; % double check needed
    end
    methods
        function obj = mobp1()
            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p) * 10.0 ;
           
           
            % init bound lower level
            obj.xl_bl = zeros(1, obj.q);
            obj.xl_bu = ones(1, obj.q) * 10;      
        end
        
        function [f, c] = evaluate_u(obj, xu, xl) 
            %-obj
            f(:, 1) = -( (xu(:, 1) + 2 * xl(:, 2) + 3)  .*  (3 * xl(:, 1) + 2) );
            f(:, 2) = -( (2 * xu(:, 1) + xl(:, 1) + 2) .* (xl(:, 2) + 1) );
          
         
            %-cie
            c(:, 1) = 3 * xu(:, 1) + xl(:, 1) + 2 * xl(:, 2) - 5;
            c(:, 2) = xl(:, 1) + xl(:, 2) - 3;
            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
           
            %-obj
            f = (xl(:, 1) + 1) .* (xu(:, 1) + xl(:, 1) + xl(:, 2) + 3);
            f = -f;
            
            %-cie
            c(:, 1) = xu(:, 1) + 2 * xl(:, 1) + xl(:, 2) - 2;
            c(:, 2) = 3 * xl(:, 1) + 2 * xl(:, 2) - 6;
          
        
         end
    end
end
