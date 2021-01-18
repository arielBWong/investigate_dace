classdef Haupt_schewefel
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
        xprime = [6.0860,6.0860];
        fprime = -13.6478;
        name = 'Haupt_schewefel';
    end
    methods
        function obj = Haupt_schewefel()
            
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);
            
            
            % init bound lower level
            obj.xl_bl = [-15, -15];
            obj.xl_bu = [15, 15];
            
        end
        function [f, con] = evaluate(obj, x)
            f=[];
            con = [];
            
        end
        
        function [f, c] = evaluate_l(obj, xu, x)
            x1 = x(:,1); x2 = x(:, 2);

            
            f = -x1 .* sin(x1 ./ 3) - 1.5 * x2 .* sin(x2 ./ 3);
            
            c = -x1 .* sin(sqrt(abs(x1))) - x2 .* sin(sqrt(abs(x2)));
            
            
        end
    end
end