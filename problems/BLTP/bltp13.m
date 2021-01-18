classdef bltp13
    properties
        p = 2;
        q = 3;
        r = 0;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        uopt = -29.2;
        lopt = 3.2;
        name =  'bltp13';
    end
    methods
        function obj = bltp13()
            % level variables
            obj.n_lvar = obj.q + obj.r;
            obj.n_uvar = obj.p + obj.r;
            
            % bounds
            %init bound upper level
            xu_bl_1 = zeros(1, obj.p);
            xu_bu_1 = ones(1, obj.p) * 1.0 ;
            xu_bl_2 = zeros(1, obj.r);
            xu_bu_2 = ones(1, obj.r) * 1.0 ;
            obj.xu_bl = [xu_bl_1, xu_bl_2];
            obj.xu_bu = [xu_bu_1, xu_bu_2];
            
            % init bound lower level
            xl_bl_1 = zeros(1, obj.q);
            xl_bu_1 = ones(1, obj.q) * 1.0;
            xl_bl_2 = zeros(1, obj.r);
            xl_bu_2 = ones(1, obj.r) * 1.0;
            obj.xl_bl = [xl_bl_1, xl_bl_2];
            obj.xl_bu = [xl_bu_1, xl_bu_2];
            
        end
        
        function [f, c] = evaluate_u(obj, xu, xl)
            xu1 = xu(:, 1 : obj.p);
            xu2 = xu(:, obj.p + 1: obj.p + obj.r);
            
            xl1 = xl(:, 1 : obj.q);
            xl2 = xl(:, obj.q+1 : obj.q+obj.r); % variable not used due to 
                                                    % class definition
                                                    % consistence and
                                                    % healthy
                                                    % laziness
            %-obj
            f =   8 * xu1(:, 1) + 4 * xu1(:, 2) - 4 * xl1(:, 1) + 40 * xl1(:, 2) + 4 * xl1(:, 3);
            f = -f;
            %-cie
            c = [];
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
            xu1 = xu(:, 1 : obj.p);
            xu2 = xu(:, obj.p + 1: obj.p + obj.r);  % variable not used due to 
                                                    % class definition
                                                    % consistence and
                                                    % healthy
                                                    % laziness
            
            xl1 = xl(:, 1 : obj.q);
            xl2 = xl(:, obj.q+1 : obj.q+obj.r);
            
            %-obj
            f = -xu1(:, 1) - 2 * xu1(:, 2) - xl1(:, 1) - xl1(:, 2) - 2 * xl1(:, 3);
            f = -f;
            
            %-cie
            c(:, 1) = -xl1(:, 1) + xl1(:, 2) + xl1(:, 3) - 1;
            c(:, 2) =  2 * xu1(:, 1) - xl1(:, 1) + 2 * xl1(:, 2) - 0.5 * xl1(:, 3) - 1;
            c(:, 3) = 2 * xu1(:,2) + 2 * xl1(:, 1) - xl1(:, 2) - 0.5*xl1(:, 3) - 1;
        
         end
    end
end
