classdef Shekel
    
    properties
        p = 1;
        q = 1;
        
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        m = 10;
        xprime;
        name;
        uopt = [];
        lopt = [];
    end
    methods
        function obj = Shekel(p, q)
            if nargin > 1
                obj.p = p;
                obj.q = q;
            end
            obj.name = 'Shekel';
            
            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            obj.xprime = ones(1, obj.q) * 4;
            
            % bounds
            %init bound upper level
            
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p) * 10.0;
            
            % init bound lower level
            obj.xl_bl = zeros(1, obj.q);
            obj.xl_bu = ones(1, obj.q) * 10.0;
            
        end
        
        function [f, c] = evaluate_u(obj, xu, xl)
            
            
            nv = 2;
            b = 0.1 * [1, 2, 2, 4, 4, 6, 3, 7, 5, 5]';
            C = [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0;
                4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6;
                4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0;
                4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6];
          
            
            outer = 0;
            for ii = 1:obj.m
                bi = b(ii);
                inner = 0;
                for jj = 1:nv
                    xj = xu(:, jj);
                    Cji = C(jj, ii);
                    inner = inner + (xj-Cji).^2;
                end
                outer = outer + 1./(inner+bi);
            end
            
            f = -outer;
            c = [];
            
            
        end
        function [f, c] = evaluate_l(obj, xu, xl)
            
            b = 0.1 * [1, 2, 2, 4, 4, 6, 3, 7, 5, 5]';
            C = [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0;
                4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6;
                4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0;
                4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6];
            
            outer = 0;
            nv = obj.q;
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
            
            f = -outer;
            c = [];
            
        end
        
        function xl_prime = get_xlprime(obj, xu)
            xl_prime = [];
        end
    end
end
