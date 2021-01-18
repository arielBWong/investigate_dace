classdef tp6
    properties
        p = 1;
        q = 1;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        xl_prime;
        name = 'tpso6';
        uopt = NaN;
        lopt = 0.0625; % double check needed
    end
    methods
        function obj = tp6(p, q)
            % level variables
             if nargin > 1
                 obj.q = q;
                 obj.p = p;
             end
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            obj.xl_prime = 0.5 * ones(1, obj.q);
            
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
            f = []
            c = [];
            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
           
            %-obj
            f = 0;
            n = 1;
           for i = 1: obj. q
               fi = 2 * sin(10 * exp(-0.2 * xl(:, i)) .* xl(:, i)) .* exp(-0.25 * xl(:, i));
               f = f+fi;
           end
           
           f = f ./ n;
           
            %-con
            c = [];
        end
        
        function plot2d_lower(obj)
            figure(1);
            xu = [];
            m = 101;
            xu1 = linspace(obj.xl_bl(1), obj.xl_bu(1), m);
            xu2 = linspace(obj.xl_bl(2), obj.xl_bu(2), m);
            [xu1, xu2] = meshgrid(xu1, xu2);
            f = zeros(m, m);
            for i = 1:m
                for j = 1:m
                    xl = [xu1(i, j), xu2(i,j)];
                    f(i, j) = obj.evaluate_l(xu, xl);
                end
            end
            surfc(xu1, xu2, f);hold on;
            colormap jet
            shading interp
            title('real landscape');
            xlabel('xl1','FontSize', 16);
            ylabel('xl2', 'FontSize', 16);
        end
    end
end
