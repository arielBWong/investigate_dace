classdef mobp5
    properties
        p = 1;
        q = 1;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'mobp5';
        uopt = NaN;
        lopt = 0; % double check needed
    end
    methods
        function obj = mobp5()
            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p) * 15.0 ;
           
           
            % init bound lower level
            obj.xl_bl = zeros(1, obj.q);
            obj.xl_bu = ones(1, obj.q) * 15;      
        end
        
        function [f, c] = evaluate_u(obj, xu, xl) 
            %-obj
            f(:, 1) = -xu - xl;
            f(:, 2) = xu.^2 + (xl-10).^2;
                 
            %-cie
            c= [];            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)          
            %-obj
            f = xl .* (xu -30);
            
            %-cie
           c = xl - xu;
        
         end
    end
end
