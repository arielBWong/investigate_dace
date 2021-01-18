classdef tp1
    properties
        p = 1;
        q = 2;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'tpso1';
        uopt = NaN;
        lopt = NaN; % double check needed
    end
    methods
        function obj = tp1()
            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);
           
           
            % init bound lower level
            obj.xl_bl = zeros(1, obj.q) ;
            obj.xl_bu = ones(1, obj.q) * 10;      
        end
        
        function [f, c] = evaluate_u(obj, xu, xl) 
            %-obj
            f(:, 1) = xl(:, 1) - xu;
            f(:, 2) = xl(:, 2);
          
         
            %-cie
            c(:, 1) = 1 + xl(:, 1) + xl(:, 2);
            c = -c;
            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
           
            %-obj
           f  = 0;
           for i = 1: obj.q
               index = xl(:, i) <= 8;
               fi = -(-8 - x(:, i))^0.1 * exp(-0.2 * (8-xl(:, i)));
               fi(index) = 0;
                f = f+  fi;
           end
            %-con
             c = [];     
         end
    end
end
