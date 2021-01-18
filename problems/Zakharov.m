classdef Zakharov
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
        fprime = 0;
        xprime =[];
        name = 'Zakharov';
    end
    methods
        function obj = Zakharov(p, q)
            if nargin > 1
                obj.q = q;
                obj.p = p;
            end
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            obj.xprime = zeros(1, obj.q);
            
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);
            
            
            % init bound lower level
            obj.xl_bl = ones(1, obj.q) * -5;
            obj.xl_bu = ones(1, obj.q) * 10 ;
            
            
            % obj.xl_bl = [-5, -5];
            % obj.xl_bu = [10, 10];
        end
        function [f, c] = evaluate_u(obj, xu, xl)
            f = [];
            c = [];
            
        end
        
        function [f, c] = evaluate_l(obj, xu, xx)
            d = size(xx, 2);
            sum1 = 0;
            sum2 = 0;
            
            for ii = 1:d
                xi = xx(:, ii);
                sum1 = sum1 + xi .^2;
                sum2 = sum2 + 0.5 .* ii .* xi;
            end
            
            f = sum1 + sum2 .^ 2 + sum2 .^ 4;
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
            surf(x1, x2, f); hold on;
            colormap jet
            shading interp
            
            
            
        end
    end
end