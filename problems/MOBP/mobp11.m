classdef mobp11
    properties
        p ;
        q ;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'mobp11';
        uopt = NaN;
        lopt = 0; % double check needed
    end
    methods
        function obj = mobp11(k)
            obj.p = k;
            obj.q = k;
            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            %init bound upper level
            obj.xu_bl = ones(1, obj.p) * -1;
            obj.xu_bu = ones(1, obj.p) * 1 ;
           
           
            % init bound lower level
            obj.xl_bl = ones(1, obj.q) * -1;
            obj.xl_bu = ones(1, obj.q) * 1;      
        end
        
        function [f, c] = evaluate_u(obj, xu, xl) 
            %-obj
           f(:, 1) = sum(exp(-xu ./ (1+ abs(xl))), 2) + sum(sin(xu ./ (1+abs(xl))), 2);
           f(:, 2) = sum(exp(-xl ./ (1 + abs(xu))), 2) + sum(sin(xl ./ (1+ abs(xu))), 2);
                 
            %-cie
            c= [];            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)          
            %-obj
            f = sum( cos(abs(xu) .* xl) ,2) + sum(sin(xu - xl), 2);
            
            %-cie
           c=xu + xl - 1;
        
         end
    end
end
