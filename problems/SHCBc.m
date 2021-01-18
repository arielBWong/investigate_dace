classdef SHCBc
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
        xprime = [1.8150,-0.8750];
        fprime = -2.9501;
        ref;
        name = 'SHCBc';
    end
    methods
        function obj = SHCBc()
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);
            
            
            % init bound lower level
            obj.xl_bl = [-2, -1];
            obj.xl_bu = [2, 1];
        end
        function [f, c] = evaluate_u(obj, xu, xl)
            f = [];
            c = [];
            
        end
        
        function [f, c] = evaluate_l(obj, xu, x)
            x1 = x(:,1); x2 = x(:, 2);
            
            f = (x1 - 2).^2 + (x2 + 1).^2 - 3;
            
            part1 = (4 - 2.1 .* x1 .^ 2 + x1 .^ 4 / 3) .* x1 .^ 2;
            part2 = x1 .* x2;
            part3 = (-4 + 4 .* x2 .^ 2) .* x2 .^ 2;
            
            c = part1 + part2 + part3;
            
        end
    end
end