classdef Shekel_curve
    
    properties
        p = 1;
        q = 0;
        r = 1;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        m = 4;
        name;
        uopt = [];
        lopt = [];
    end
    methods
        function obj = Shekel_curve(m)
            if nargin == 1
                obj.m = m;
            end
            obj.name = 'Shekel_curve';
            
            % level variables
            obj.n_lvar = obj.q + obj.r;
            obj.n_uvar = obj.p + obj.r;
            
            % bounds
            %init bound upper level
            
            obj.xu_bl = ones(1, obj.p+obj.r) * -10.0;
            obj.xu_bu = ones(1, obj.p+obj.r) * 10.0;
            
            % init bound lower level
            obj.xl_bl = ones(1, obj.q + obj.r) * -5;
            obj.xl_bu = ones(1, obj.q + obj.r) * 5;
            
        end
        
        function [f, c] = evaluate_u(obj, xu, xl)     
            
            f = [];
            c = [];
            
            
        end
        function [f, c] = evaluate_l(obj, xu, xl)
            
%             b = 0.1 * [1, 2, 2, 4, 4, 6, 3, 7, 5, 5]';
%             C = [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0;
%                 4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6;
%                 4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0;
%                 4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6];
            
           b = [0.05, 0.2, 0.3, 0.3]';
           C = [-4, 4, -2, 2;
                -4, 4, -2, 2;
                -4, 4, -2, 2;
                -4, 4, -2, 2];
            
            outer = 0;
            nv = 1;
            for ii = 1:obj.m
                bi = b(ii);
                inner = 0;
                for jj = 1:nv
                    xj = xl(:, jj);
                    Cji = C(jj, ii);
                    inner = inner + (xj-Cji).^2;
                end
                outer = outer + 1./(inner+bi);
            end
            outer = outer - sum(xl.^2, 2);
            f = -outer;
            c = [];
            
        end
        
        function xl_prime = get_xlprime(obj, xu)
            xl_prime = [];
        end
    end
end
