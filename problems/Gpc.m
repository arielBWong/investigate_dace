classdef Gpc
    properties
        p = 1;
        q = 2;
        n_lvar;
        n_uvar;
        n_con;
        n_obj;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        ref;
        fprime = 5.6692;
        xprime = [0.5955,-0.4045];
        name = 'Gpc';
    end
    methods
        function obj = Gpc()
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);
            
            
            % init bound lower level
            obj.xl_bl = [-2, -2];
            obj.xl_bu = [2, 2];

        end
        function [f, con] = evaluate(obj, x)
            x1 = x(:,1); x2 = x(:, 2);
            
            A = 19 - 14 * x1 + 3 * x1.^2 - 14 * x2 + 6 * x1 .* x2 + 3 * x2  .^ 2;
            B = 18 - 32 * x1 + 12 * x1.^2 + 48 * x2 - 36 * x1 .* x2 + 27 * x2 .^ 2;
            
            f = (1 + A .* (x1 + x2 + 1).^2) * (30 + B .* (2 * x1 - 3 * x2).^2);
            f = log(f);
            
            g1 = -3 * x1 + (-3 * x2).^3;
            g2 = x1 - x2 - 1;
            
            con = [g1, g2];
            
        end
        
        function [f, con] = evaluate_l(obj, xu, x)
            x1 = x(:,1); x2 = x(:, 2);
            
            A = 19 - 14 * x1 + 3 * x1.^2 - 14 * x2 + 6 * x1 .* x2 + 3 * x2  .^ 2;
            B = 18 - 32 * x1 + 12 * x1.^2 + 48 * x2 - 36 * x1 .* x2 + 27 * x2 .^ 2;
            
            f = (1 + A .* (x1 + x2 + 1).^2) .* (30 + B .* (2 * x1 - 3 * x2).^2);
            
            f = log(f);
            
            g1 = -3 * x1 + (-3 * x2).^3;
            g2 = x1 - x2 - 1;
            
            con = [g1, g2];
            
        end
        
         function c = cons1(obj, x1, x2)
            c = -3 * x1 + (-3 * x2).^3;
         end
         
          function c = cons2(obj, x1, x2)         
             c = x1 - x2 - 1;
         end

    end
end