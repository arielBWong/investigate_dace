classdef mobp10
    properties
        p = 5;
        q = 1;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'mobp10';
        uopt = NaN;
        lopt = 0; % double check needed
    end
    methods
        function obj = mobp10()
            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            %init bound upper level
            xu1_bl = ones(1, 1) * (-1);
            xu1_bu = ones(1, 1);
            xu2_bl  = ones(1, obj.p-1) * (-5);
            xu2_bu = ones(1, obj.p-1) * 5;
            
            obj.xu_bl = [xu1_bl, xu2_bl];
            obj.xu_bu = [xu1_bu, xu2_bu] ;        
           
            % init bound lower level
            obj.xl_bl = ones(1, obj.q) * 1;
            obj.xl_bu = ones(1, obj.q) * 2;      
        end
        
        function [f, c] = evaluate_u(obj, xu, xl) 
            %-obj
            f(:, 1) =(1 - xu(:, 1)) .* (1 + xu(:, 2) .^ 2 + xu(:, 3) .^2)  .* xl;
            f(:, 2) = xu(:, 1) .* (1 + xu(:, 2) .^2 + xu(:, 3) .^2 ) .* xl;
                 
            %-cie
            c= [];            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)          
            %-obj
            f = (1-xu(:, 1)) .* (1 + xu(:, 4) .^ 2 + xu(:, 5) .^ 2) .* xl;
            
            %-cie
           c= (1 - xu(:, 1)) .* xl + 0.5 * xu(:, 1) .* xl - 1;
           c = -c;
           
                     
           
           
           
           
           
           
        
         end
    end
end
