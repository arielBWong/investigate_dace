classdef mobp2
    properties
        p = 1;
        q = 1;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'mobp2';
        uopt = NaN;
        lopt = 0; % double check needed
    end
    methods
        function obj = mobp2()
            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p) * 10.0 ;
            
            
            % init bound lower level
            obj.xl_bl = zeros(1, obj.q);
            obj.xl_bu = ones(1, obj.q) * 10;
        end
        
        function [f, c] = evaluate_u(obj, xu, xl)
            %-obj
            f(:, 1) = -(-2 * xu);
            f(:, 2) = -(-xu + 5 * xl);
            %-cie
            c = [];
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
            %-obj
            f = -xl;
            
            %-cie
            c(:, 1) = xu - 2 * xl - 4;
            c(:, 2) = 2 * xu - xl - 24;
            c(:, 3) = 3 * xu + 4 * xl - 96;
            c(:, 4) = xu + 7 * xl - 126;
            c(:, 5) = -4 * xu + 5 * xl - 65;
            c(:, 6) = -(xu + 4 * xl - 8);
        end
    end
end
