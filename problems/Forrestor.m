classdef Forrestor
    properties
        p = 1;
        q = 1;
        n_lvar;
        n_uvar;
        n_con;
        n_obj;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        ref;
        xprime = [];
        fprime = [];
        name = 'Forrestor';
        
    end
    methods
        function obj = Forrestor()
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);
            
            
            % init bound lower level
            obj.xl_bl = 0;
            obj.xl_bu = 1;
        end
        function [f, con] = evaluate(obj, x)
            f = [];
            con = [];
            
        end
        
        function [f, con] = evaluate_l(obj, xu, x)
            f = (6 .* x - 2).^2 .* sin(12 .* x -4);
            con = [];
            
        end
    end
end