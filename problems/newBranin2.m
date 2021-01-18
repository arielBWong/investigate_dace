classdef newBranin2
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
        fprime =-243.0747;
        xprime = [3.2143,0.9633];
        name = 'newBranin2';
    end
    methods
        function obj = newBranin2()
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);
            
            
            % init bound lower level
            obj.xl_bl = [-5, 0];
            obj.xl_bu = [10, 15];
        end
        function [f, c] = evaluate_u(obj, xu, xl)
            f = [];
            c = [];
            
        end
        
        function [f, c] = evaluate_l(obj, xu, x)
            x1 = x(:,1); x2 = x(:, 2);
            
            % objective
            f = -(x1 - 10.) .^ 2 - (x2 - 15.) .^ 2;
            
            a = 1.0;
            b = 5.1 / (4 * pi ^ 2);
            c = 5.0 / pi;
            r = 6.0;
            s = 10.0;
            t = 1.0 / (8.0 * pi);
            
            part1 = a .* (x2 - b .* x1 .^ 2 + c .* x1 - 6.0) .^ 2.0;
            part2 = s .* (1 - t) .* cos(x1);
            part3 = s;
            
            % constraint
            c = part1 + part2 + part3 - 2;
            
        end
        
        
        function c = cons(obj, x1, x2)
            
           
            a = 1.0;
            b = 5.1 / (4 * pi ^ 2);
            c = 5.0 / pi;
            r = 6.0;
            s = 10.0;
            t = 1.0 / (8.0 * pi);
            
            part1 = a .* (x2 - b .* x1 .^ 2 + c .* x1 - 6.0) .^ 2.0;
            part2 = s .* (1 - t) .* cos(x1);
            part3 = s;
            
            % constraint
            c = part1 + part2 + part3 - 2;
            
        end
    end
end