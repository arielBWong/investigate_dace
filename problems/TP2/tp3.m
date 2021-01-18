classdef tp3
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
        name = 'tpso3';
        uopt = NaN;
        lopt = NaN; % double check needed
    end
    methods
        function obj = tp3(p, q)
            % level variables
             if nargin > 1
                 obj.q = q;
                 obj.p = p;
             end
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            obj.xprime = 0.1 * ones(1, obj.q);
            
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
               index  = xl(:, i) > 0.4 & xl(:, i) <= 0.6;
               other = ~index;
               
               fb1  = exp(-2 * log(2) *( ( (xl(:, i)-0.1)/ 0.8).^2) ) .* sqrt(sin( abs(5 * pi * xl(:, i))) ); 
               fb2  = exp(-2 * log(2) * (( (xl(:, i)-0.1)/ 0.8).^2) ) .* sin( abs(5 * pi * xl(:, i))) .^ 6; 
               
               fb1(other) = 0;
               fb2(index) = 0;
               f = f + fb1 + fb2;
           end
           
           f = f ./ n;
           f = -f;
            %-con
            c = [];
         end
    end
end
