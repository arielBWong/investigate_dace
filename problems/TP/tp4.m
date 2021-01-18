classdef tp4
    properties
        p =2;     % typo problem  in paper
        q = 3;    % typo problem in paper
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'tp4';
        uopt = NaN;
        lopt = NaN;  % double check needed
    end
    methods
        function obj = tp4()
            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p) * 100;  % undefined in paper
           
           
            % init bound lower level
            obj.xl_bl = zeros(1, obj.q);
            obj.xl_bu = ones(1, obj.q) * 100;        % undefined in paper
        end
        
        function [f, c] = evaluate_u(obj, xu, xl) 
            %-obj
            f(:, 1) = xu(:, 1) + 9 * xu(:, 2) + 10 *  xl(:, 1) + xl(:, 2) + 3 *  xl(:, 3);
            f(:, 2) = 9 * xu(:, 1) + 2 * xu(:, 2) + 2 *  xl(:, 1) + 7 * xl(:, 2) + 4 *  xl(:, 3);
          
         
            %-cie
            c(:, 1) = 3 * xu(:, 1) + 9 * xu(:, 2) + 9 *  xl(:, 1) + 5 * xl(:, 2) + 3 *  xl(:, 3) - 1039;
            c(:, 2) = (-4) *  xu(:, 1) - xu(:, 2) + 3 *  xl(:, 1) - 3 * xl(:, 2) + 2 *  xl(:, 3) - 94;
            
            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
           
            %-obj
            f(:, 1) = 4*  xu(:, 1) + 6 * xu(:, 2) + 7 *  xl(:, 1) + 4 * xl(:, 2) + 8 *  xl(:, 3);
            f(:, 2) = 6 *  xu(:, 1) + 4 * xu(:, 2) + 8 *  xl(:, 1) +7 * xl(:, 2) + 4 *  xl(:, 3);
            f = sum(f, 2);
            
            %-cie
            c(:, 1) = 3 * xu(:, 1) - 9 * xu(:, 2) - 9 *  xl(:, 1) - 4 * xl(:, 2) + 0 *  xl(:, 3) - 61;
            c(:, 2) =  5 * xu(:, 1) + 9 * xu(:, 2) + 10 *  xl(:, 1) - xl(:, 2) - 2 *  xl(:, 3) - 924;
            c(:, 3) =  3 * xu(:, 1) - 3 * xu(:, 2) + 0 *  xl(:, 1) + xl(:, 2) + 5 *  xl(:, 3) - 420;                  
                  
         end
    end
end
