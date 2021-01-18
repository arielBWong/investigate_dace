classdef tp5
    properties
        p = 1;
        q = 1;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        xprime;
        name = 'tpso5';
        uopt = NaN;
        lopt = NaN; % double check needed
    end
    methods
        function obj = tp5(p, q)
            % level variables
             if nargin > 1
                 obj.q = q;
                 obj.p = p;
             end
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            obj.xprime = 0.5 * ones(1, obj.q);
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);
           
           
            % init bound lower level
            obj.xl_bl = zeros(1, obj.q) ;
            obj.xl_bu = ones(1, obj.q) * 1;      
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
               index1  = xl(:, i) <  0.4696;
               index2  = xl(:, i) >= 0.4696 & xl(:, i) <= 0.5304;
               index3  = xl(:, 1) > 0.5304;
               
               fb1  =  -0.5 * exp(-0.5 * ((xl(:, i) - 0.4).^2 ./ (0.05^2)));
               fb2  =  -0.6 * exp(-0.5 * ((xl(:, i) - 0.5).^2 ./ (0.02^2)));
               fb3  =  -0.5 * exp(-0.5 * ((xl(:, i) - 0.6).^2 ./ (0.05^2)));
               
               fb1(index2) = 0; fb1(index3) = 0;
               fb2(index1) = 0; fb2(index3) = 0;
               fb3(index1) = 0; fb3(index2) = 0;
               f = f+ fb1 + fb2 + fb3;
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
