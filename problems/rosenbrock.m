classdef  rosenbrock
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
        name = 'rosenbrock';
    end
    methods
        function obj = rosenbrock(p, q)
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
            obj.xl_bl = -5 * ones(1, obj.q);
            obj.xl_bu = 10 * ones(1, obj.q);
        end
        function [f, c] = evaluate_u(obj, xu, xl)
            f = [];
            c = [];
            
        end
        
        function [f, c] = evaluate_l(obj, xu, xx)
            d = size(xx, 2);
            
  
            sum = 0;
            for ii = 1:(d-1)
                xi = xx(:, ii);
                xnext = xx(:, ii+1);
                new = 100*(xnext - xi .^ 2) .^ 2 + (xi-1) .^ 2;
                sum = sum + new;
            end
            
            f = sum;
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
            surfc(x1, x2, f);
            colormap jet
            shading interp
            
        end
    end
end