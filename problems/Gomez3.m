classdef Gomez3
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
        xprime = [0.1093,-0.6234];
        fprime = -0.9711;
        name = 'Gomez3';
    end
    methods
        function obj = Gomez3()
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);
            
            
            % init bound lower level
            obj.xl_bl = [-1, -1];
            obj.xl_bu = [1, 1];
        end
        function [f, c] = evaluate_u(obj, xu, xl)
            f = [];
            c = [];
            
        end
        
        function [f, c] = evaluate_l(obj, xu, x)
            x1 = x(:,1); x2 = x(:, 2);
            
            f = (4 - 2.1 .* x1 .^ 2 + (x1 .^ 4)./3) .* x1 .^ 2 + ...
                    x1 .* x2 + ...
                    (-4 + 4 .* x2 .^ 2) .* x2 .^2;
            c = -sin(4 * pi .* x1) + 2 * (sin(2 * pi .* x2)) .^ 2;

        end
    end
end