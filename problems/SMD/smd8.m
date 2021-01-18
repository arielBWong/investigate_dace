classdef smd8
    properties
        p = 1;
        q = 2;
        r = 1;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name;
        uopt = 0;
        lopt = 0;
        xu_prime = [0, 0];
        xl_prime = [1, 1, 0];
    end
    methods
        function obj = smd8(p, q, r)
            if nargin == 3
                obj.p = p;
                obj.q = q;
                obj.r = r;
            end
            obj.name = 'SMD8';
            
            % level variables
            obj.n_lvar = obj.q + obj.r;
            obj.n_uvar = obj.p + obj.r;
            
            % bounds
            %init bound upper level
            xu_bl_1 = ones(1, obj.p)* (-5.0);
            xu_bu_1 = ones(1, obj.p) * 10.0;
            xu_bl_2 = ones(1, obj.r)* (-5.0);
            xu_bu_2 = ones(1, obj.r) * 10.0;
            obj.xu_bl = [xu_bl_1, xu_bl_2];
            obj.xu_bu = [xu_bu_1, xu_bu_2];
            
            % init bound lower level
            xl_bl_1 = ones(1, obj.q) * (-5.0);
            xl_bu_1 = ones(1, obj.q) * 10.0;
            xl_bl_2 = ones(1, obj.r) * (-5.0);
            xl_bu_2 = ones(1, obj.r) * 10.0;
            obj.xl_bl = [xl_bl_1, xl_bl_2];
            obj.xl_bu = [xl_bu_1, xl_bu_2];
            
        end
        
        function [f, c] = evaluate_u(obj, xu, xl)
            xu1 = xu(:, 1 : obj.p);
            xu2 = xu(:, obj.p + 1: obj.p + obj.r);
            
            xl1 = xl(:, 1 : obj.q);
            xl2 = xl(:, obj.q+1 : obj.q+obj.r);
            
            term2 = 0;
            for i=1:obj.q-1
                term2 = term2 + (xl1(:, i+1) - xl1(:, i).^2).^2 + (xl1(:, i) - 1).^2;
            end
            
            f = 20 + exp(1)- 20*exp(-0.2*sqrt(1/obj.p * sum((xu1).^2, 2))) ...
                - exp(1/obj.p * sum(cos(2*pi*xu1), 2))  ...
                - term2 ...
                + sum((xu2).^2, 2) - sum((xu2 - xl2.^3).^2, 2);
            
            c = [];
            
        end
        function [f, c] = evaluate_l(obj, xu, xl)
            xu1 = xu(:, 1 : obj.p);
            xu2 = xu(:, obj.p + 1: obj.p + obj.r);
            
            xl1 = xl(:, 1 : obj.q);
            xl2 = xl(:, obj.q+1 : obj.q+obj.r);
            
            term2 = 0;
            for i=1:obj.q-1
                term2 = term2 + (xl1(:, i+1) - xl1(:, i).^2).^2 + (xl1(:,i) - 1).^2;
            end
            %-obj
            f = sum(abs(xu1), 2) ...
                + term2 ...
                + sum((xu2 - xl2.^3).^2, 2);
            %-cie 
            c = [];
            
        end
        
        function xl_prime = get_xlprime(obj, xu)
            for i = 1:obj.q
                xl_prime(i) = 0;
            end
            
            j = 1;
            for i = obj.q + 1 : obj.q + obj.r
      
                xl_prime(i) = xu(obj.p+ j)^(1/3);
                j = j + 1;
            end
        end
    end
end
