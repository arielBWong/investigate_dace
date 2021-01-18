classdef bltp5
    properties
        p = 1;
        q = 2;
        r = 0;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        uopt = -1.4074;
        lopt =  7.6172;
        name =  'bltp5';
    end
    methods
        function obj = bltp5()
            % level variables
            obj.n_lvar = obj.q + obj.r;
            obj.n_uvar = obj.p + obj.r;
            
            % bounds
            %init bound upper level
            xu_bl_1 = zeros(1, obj.p);
            xu_bu_1 = ones(1, obj.p) * 4.0 ;
            xu_bl_2 = zeros(1, obj.r);
            xu_bu_2 = ones(1, obj.r) * 4.0 ;
            obj.xu_bl = [xu_bl_1, xu_bl_2];
            obj.xu_bu = [xu_bu_1, xu_bu_2];
            
            % init bound lower level
            xl_bl_1 = zeros(1, obj.q);
            xl_bu_1 = ones(1, obj.q) * 2.0;
            xl_bl_2 = zeros(1, obj.r);
            xl_bu_2 = ones(1, obj.r) * 2.0;
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
            f = (xu1 - 1) .^ 2 + 2 * xl1(:, 1) .^ 2 - 2 * xu1;
         
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
            f = (2 * xl1(:, 1) - 4).^ 2 + (2 * xl1(:, 2) - 1).^ 2 + xu1 .* xl1(:, 1);
            
            %-cie
            c(:, 1) = 4 * xu1 + 5 * xl1(:,1) + 4*xl1(:, 2) - 12;
            c(:, 2) = -4 * xu1-5*xl1(:,1)+4*xl1(:, 2) + 4;
            c(:, 3) = 4 * xu1-4*xl1(:, 1)+5*xl1(:, 2) - 4;
            c(:, 4) = -4 * xu1 + 4 * xl1(:, 1) + 5 * xl1(:, 2)-4;
         end
    end
end
