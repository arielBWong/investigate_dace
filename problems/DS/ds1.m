classdef ds1
    properties
        p ;
        q;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'ds1';
        uopt = NaN;
        lopt = NaN; % double check needed
    end
    methods
        function obj = ds1(k)
            obj.p = k;
            obj.q = k;
            
            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            % init bound upper level
            obj.xu_bl = [1, ones(1, k-1) * (-k)];
            obj.xu_bu = [4, ones(1, k-1) * k];
            
            
            % init bound lower level
            obj.xl_bl = ones(1, obj.q) * (-k);
            obj.xl_bu = ones(1, obj.q) * k;
        end
        
        function [f, c] = evaluate_u(obj, xu, xl)
            %-obj
            r = 0.1;
            alpha = 1;
            gamma = 1;
            tao = 1;
            
            p3 = tao* sum((xl(:, 2:obj.n_lvar) - xu(:, 2:obj.n_uvar)) .^ 2, 2);
            
            p2 = 2: obj.n_lvar;
            p2 =( p2 - 1) /2;
            p2 =  sum((xu(:, 2:obj.n_lvar) - p2) .^2 , 2);
            f(:, 1) = (1+ r -cos(alpha * pi * xu(:, 1))) + p2 + p3 - r * cos(gamma * pi/2 * xl(:, 1) ./ xu(:, 1)) ;
            
            f(:, 2) =  (1+ r - sin(alpha * pi * xu(:, 1))) + p2 + p3 - r * sin(gamma * pi/2 * xl(:, 1) ./ xu(:, 1)) ;
            
            
            %-cie
            c = [];
            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
            p2 = sum(( xl(:, 2: obj.n_lvar) - xu(:, 2: obj.n_uvar)) .^2, 2);
            
            %-obj
            p3 = 10 * (1 - cos(pi/obj.n_lvar .* (xl(:, 2:obj.n_lvar) - xu(:, 2:obj.n_uvar))));
            f(:, 1) = xl(:, 1) .^2 + p2 + sum(p3  ,2);
            
            p3 = abs(sin(pi/obj.n_lvar .* (xl(:, 2:obj.n_lvar) - xu(:, 2:obj.n_uvar))));
            p2 = sum(( xl - xu) .^2, 2);
            f(:, 2) =  p2 + sum(p3, 2);
            f = sum(f, 2);
            
            %-cie
            c = [];
            
        end
    end
end
