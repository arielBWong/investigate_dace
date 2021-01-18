classdef BNH
    properties       
        n_var;
        xl;
        xu;
        n_con;
        n_obj;
        ref;
        name;
        %---
         p = 1;
        q = 2;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
    end
    methods
        function obj = BNH()
            obj.n_var = 2;
            obj.xl = [0, 0];
            obj.xu = [5, 3];
            obj.n_con = 2;
            obj.n_obj = 2;
            obj.ref = [140, 50];
            obj.name = 'BNH';
             %---bl adapt
            obj.n_lvar = obj.q;
            obj.xl_bl = [0, 0];
            obj.xl_bu =   [5, 3];
            %-----
        end
        function [f, con] = evaluate(obj, x)
            % ----------------------------------------------------------------------------
            % the constrains are inactive
            % Deb, K. Multi-objective optimization using evolutionary algorithms.
            % John Wiley & Sons, 2001.
            % x1 = [0,5]; x2 = [0,3].
            % ----------------------------------------------------------------------------
            x1 = x(:,1); x2 = x(:, 2);
            f(:, 1) = 4*x1.^2 + 4*x2.^2;
            f(:, 2) =(x1 - 5).^2 + (x2 - 5).^2;
            con(:, 1) = (x1 - 5).^2 + x2.^2 -25;
            con(:, 2) = 7.7 - (x1 - 8).^2 - (x2 + 3).^2;           
        end
        function [f, con] = evaluate_l(obj,xu,  x)
            % ----------------------------------------------------------------------------
            % the constrains are inactive
            % Deb, K. Multi-objective optimization using evolutionary algorithms.
            % John Wiley & Sons, 2001.
            % x1 = [0,5]; x2 = [0,3].
            % ----------------------------------------------------------------------------
            x1 = x(:,1); x2 = x(:, 2);
            f(:, 1) = 4*x1.^2 + 4*x2.^2;
            f(:, 2) =(x1 - 5).^2 + (x2 - 5).^2;
            con(:, 1) = (x1 - 5).^2 + x2.^2 -25;
            con(:, 2) = 7.7 - (x1 - 8).^2 - (x2 + 3).^2;           
        end
    end
end