classdef rastrigin
    properties
        p = 2;
        q = 2;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        xprime;
        name = 'rastrigin';

    end
    methods
        function obj = rastrigin(p, q)
            % level variables
             if nargin > 1
                 obj.q = q;
                 obj.p = p;
             end
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            obj.xprime = zeros(1, obj.q);
            
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);
           
           
            % init bound lower level
            obj.xl_bl = ones(1, obj.q)  * -5.12;
            % obj.xl_bl = ones(1, obj.q)  * 0;
            obj.xl_bu = ones(1, obj.q) * 5.12;      
        end
        
        function [f, c] = evaluate_u(obj, xu, xl) 
            %-obj
            f = [];
            c = [];
            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
           
            %-obj
            f = 0;
            n = 1;
           for i = 1: obj. q
               fb = xl(:, i).^2 - 10 * cos(2 * pi * xl(:, i));
               f = f+ fb ;
           end
           
         f = 10 * obj.q + f;
            %-con
            c = [];
         end
    end
end
