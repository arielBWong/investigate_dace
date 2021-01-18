classdef ds4
    properties
        p ;
        k;
        L;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'ds4';
        uopt = NaN;
        lopt = NaN;                                 % double check needed
    end
    methods
        function obj = ds4(k, L)
            obj.p = 1;
            obj.k = k;
            obj.L = L;
            
            % level variables
            obj.n_lvar = obj.k + obj.L + 1;
            obj.n_uvar = obj.p;
            
            % bounds
            % init bound upper level
            obj.xu_bl =ones(1, 1) ;
            obj.xu_bu =ones(1, 1)  * 2;            
            
            % init bound lower level
            obj.xl_bl = [-1,  ones(1, obj.n_lvar-1) * (-k-L) ];
            obj.xl_bu = [1, ones(1, obj.n_lvar -1) * (k+L) ];
        end
        
        function [f, c] = evaluate_u(obj, xu, xl)
            %-obj
           
            pk = xl(:, 2: obj.k+1) .^ 2;
           
            
            f(:, 1) =  (1 - xl(:, 1)) .* (1+ sum(pk, 2)) .* xu(:, 1);  
            f(:, 2) = xl(:, 1)  .* (1 + sum(pk, 2)) .* xu(:, 1);
           
            %-cie
            c = (1 - xl(:, 1)) .*  xu(:, 1) + 1/2 * xl(:, 1) .*  xu(:, 1) - 1;
            c = -c;           
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
          
           pL =  xl(:, obj.k+1 + 1: obj.k +obj.L + 1) .^2;
           f(:, 1) =  (1 - xl(:, 1)) .* (1+ sum(pL, 2)) .* xu(:, 1);  
            f(:, 2) = xl(:, 1)  .* (1 + sum(pL, 2)) .* xu(:, 1);
           
            f = sum(f, 2);
            
            %-cie
            c =[];
            
        end
    end
end
