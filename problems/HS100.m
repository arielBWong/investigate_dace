classdef HS100
    properties
        p = 1;
        q = 7;
        n_lvar;
        n_uvar;
        n_con;
        n_obj;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        ref;
        name = 'HS100';
    end
    methods
        function obj = HS100()
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);
            
            
            % init bound lower level
            obj.xl_bl = ones(1, obj.q) * -5;
            obj.xl_bu =  ones(1, obj.q) * 5;
            
        end
        function [f, con] = evaluate(obj, x)
            f = [];
            con = [];
            
        end
        
        function [f, c] = evaluate_l(obj, xu, x)
            x1 = x(:,1); x2 = x(:, 2);
            x3 = x(:,3); x4 = x(:, 4);
            x5 = x(:,5); x6 = x(:, 6);
            x7 = x(:,7);
            
            f = (x1 - 10).^2 + ...
                5 .* (x2 - 12) .^ 2 + ...
                x3 .^ 4 + ...
                3 .* (x4 - 11) .^2 + ...
                10 .* x5 .^ 6 + ...
                7 .* x6 .^ 2 + ...
                x7 .^ 4 - ...
                4 .* x6 .* x7 - ...
                10 .* x6 - ...
                8 .* x7;
            
            g1 = 127 - 2 .* x1 .^ 2 - 3 .* x2 .^ 4 - ...
                x3 - 4 .* x4 .^ 2 - 5 .* x5;
            g1 = -g1;
            
            g2 = -4 .* x1 .^ 2 - x2 .^ 2 + 3 .* x1 .* x2 - 2 .* x3 .^ 2 - 5 .* x6 + 11 .* x7;
            g2 = -g2;
            
            c = g1;
            
        end
    end
end