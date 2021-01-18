classdef Camelback
    properties
        p = 2;
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
        fprime = -1.0316;
        xprime = [0.0898, -0.7126];
        name = 'Camelback';
    end
    methods
        function obj = Camelback(p, q)
             if nargin > 1
                 obj.q = q;
                 obj.p = p;
             end
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
            x1 = x(:, 1);
            x2 = x(:, 2);
            
            term1 = (4 - 2.1 .* x1 .^ 2 + (x1 .^ 4) ./ 3) .* x1 .^ 2;
            term2 = x1 .* x2;
            term3 = (-4 + 4 .* x2 .^2) .* x2 .^ 2;
            
            f = term1 + term2 + term3;
            
            % constraint
            c = [];
            
        end
        
        function plot_2d(obj)
            m = 101;
            x1 = linspace(obj.xl_bl(1), obj.xl_bu(1), m);
            x2 = linspace(obj.xl_bl(2), obj.xl_bu(2), m);
            [x1, x2] = meshgrid(x1, x2);
            f = zeros(m, m);
            
            for i = 1:m
                for j = 1:m                   
                    f(i, j) = evaluate_l(obj, [], [x1(i, j), x2(i, j)]);
                end
            end
            surf(x1, x2, f);
        end
        

    end
end