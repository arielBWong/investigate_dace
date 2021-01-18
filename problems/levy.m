classdef  levy
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
        fprime = 0 ;
        xprime = [];
        name = 'levy';
    end
    methods
        function obj = levy(p, q)
             if nargin > 1
                 obj.q = q;
                 obj.p = p;
             end
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            obj.xprime = ones(1, obj.q);
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);
            
            
            % init bound lower level
            obj.xl_bl = -10 * ones(1, obj.q);
            obj.xl_bu = 10 * ones(1, obj.q);
        end
        function [f, c] = evaluate_u(obj, xu, xl)
            f = [];
            c = [];
            
        end
        
        function [f, c] = evaluate_l(obj, xu, x)
            d = size(x, 2);
            
            for ii = 1:d
                w(:, ii) = 1 + (x(:, ii) - 1) ./ 4;
            end
            
            term1 = (sin(pi  .* w(:, 1))) .^2;
            term3 = (w(:, d)-1) .^ 2  .*  (1 + (sin(2 .* pi .*  w(:, d))) .^2);
            
            sum = 0;
            for ii = 1:(d-1)
                wi = w(:, ii);
                new = (wi - 1) .^ 2 .*  (1+10 .* (sin(pi .* wi + 1)) .^ 2);
                sum = sum + new;
            end
            
            f = term1 + sum + term3;
            c = [];
        end
        
        
        function plot_problem(obj)
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