classdef ds2
    properties
        p ;
        q;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'ds2';
        uopt = NaN;
        lopt = NaN; % double check needed
    end
    methods
        function obj = ds2(k)
            obj.p = k;
            obj.q = k;

            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            % init bound upper level
            obj.xu_bl = [0.001, ones(1, k-1) * (-k)];
            obj.xu_bu = [k, ones(1, k-1) * k];
           
           
            % init bound lower level
            obj.xl_bl = ones(1, obj.q) * (-k);
            obj.xl_bu = ones(1, obj.q) * k;      
        end
        
        function [f, c] = evaluate_u(obj, xu, xl) 
            %-obj
            r = 0.25;
            tao = 1 ;
            gamma = 1;
            %---
            v1 = zeros(size(xu, 1), 1);
            for i = 1: size(xu, 1)
                if xu(i, 1) > 1
                    v1(i, 1) =  xu(i, 1) - (1 - cos(0.2* pi));
                else
                    v1(i, 1) = cos(0.2* pi) * xu(i, 1) + sin(0.2*pi) * sqrt(abs(0.02 * sin(5 * pi * xu(i, 1))));
                end 
            end
            %----
             v2 = zeros(size(xu, 1), 1);
             for i = 1: size(xu, 1)
                if xu(i, 1) > 1
                    v2(i, 1) = 0.1 * (xu(i, 1) - 1) - sin(0.2 * pi);
                else
                    v2(i, 1) =  -sin(0.2*pi ) * xu(i, 1) + cos(0.2*pi) * sqrt(abs(0.02 * sin(5* pi * xu(i, 1))));
                end 
            end
            %---  
            p2 = xu(:, 2: obj.n_uvar) .^ 2 + 10 * (1 - cos(pi/obj.n_uvar .* xu(:, 2: obj.n_uvar)));
            p2 = sum(p2, 2);
            p3 = (xu(:, 2: obj.n_uvar) - xl(:, 2:obj.n_lvar)) .^ 2;
            p3 = tao * sum(p3, 2);
            p4 = gamma * pi / 2 * xl(:, 1) ./ xu(:, 1);
            f(:, 1) = v1 +    p2 + p3 + r* cos(p4);
            f(:, 2) = v2 + p2 + p3 + r* sin(p4);
         
            %-cie
          c = [];
            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
        
           
            %-obj
          
            f(:, 1) = xl(:, 1) .^ 2 + sum((xl(:, 2: obj.n_lvar) - xu(:, obj.n_uvar)) .^ 2, 2);
            
            p1 = 1:obj.n_lvar;
            f(:, 2) =  (xl - xu) .^ 2 * p1';  % dot product
            
            
           
            f = sum(f, 2);
            
            %-cie
            c = [];
                  
         end
    end
end
