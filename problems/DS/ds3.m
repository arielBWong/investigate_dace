classdef ds3
    properties
        p ;
        q;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'ds3';
        uopt = NaN;
        lopt = NaN;                                 % double check needed
    end
    methods
        function obj = ds3(k)
            obj.p = k;
            obj.q = k;
            
            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            % init bound upper level
            obj.xu_bl =zeros(1, k) ;
            obj.xu_bu =ones(1, k)  * k;            
            
            % init bound lower level
            obj.xl_bl = ones(1, obj.q) * (-k);
            obj.xl_bu = ones(1, obj.q) * k;
        end
        
        function [f, c] = evaluate_u(obj, xu, xl)
            %-obj
            tao = 1;
            %----
            m =3: obj.n_uvar;
            m = m/2;
            p2 = sum((xu(:, 3: obj.n_uvar) -  m) .^2, 2);
            %---
            m = xl(:, 3: obj.n_lvar) - xu(:, 3: obj.n_uvar);
            p3 = tao*  sum(m.^2, 2);
            %---
            R = 0.1 + 0.15 * abs(sin(2* pi *(xu(:, 1) - 0.1)));
            
            f(:, 1) = xu(:, 1) + p2 + p3 - R .* cos(4 * atan((xu(:, 2) - xl(:, 2)) ./ (xu(:, 1) - xl(:, 1))));       
            f(:, 2) = xu(:, 2) + p2 + p3 - R .* sin(4 * atan((xu(:, 2) - xl(:, 2)) ./ (xu(:, 1) - xl(:, 1))));
           
            %-cie
            c = xu(:, 2) - (1- xu(:, 1) .^ 2);
            c = -c;           
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
            r = 0.2;
            %---
            m = xl(:, 3: obj.n_lvar) - xu(:, 3: obj.n_uvar);
            p3 =sum(m.^2, 2);
            
           f(:, 1) = xl(:, 1) +  p3;
           f(:, 2) = xl(:, 2) + p3;
            f = sum(f, 2);
            
            %-cie
            c = (xl(:, 1) - xu(:, 1)) .^2 + (xl(:, 2) - xu(:, 2)) - r.^2;
            
        end
    end
end
