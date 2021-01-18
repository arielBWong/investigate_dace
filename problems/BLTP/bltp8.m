classdef bltp8
    properties
        p = 1;
        q = 1;
        r = 1;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        uopt = -1;
        lopt =  0;
        name =  'bltp8';
    end
    methods
        function obj = bltp8()
            % level variables
            obj.n_lvar = obj.q + obj.r;
            obj.n_uvar = obj.p + obj.r;
            
            % bounds
            %init bound upper level
            xu_bl_1 = zeros(1, obj.p);
            xu_bu_1 = ones(1, obj.p) * 1.5 ;
            xu_bl_2 = zeros(1, obj.r);
            xu_bu_2 = ones(1, obj.r) * 1.5 ;
            obj.xu_bl = [xu_bl_1, xu_bl_2];
            obj.xu_bu = [xu_bu_1, xu_bu_2];
            
            % init bound lower level
            xl_bl_1 = ones(1, obj.q) * 0.5;
            xl_bu_1 = ones(1, obj.q) * 1.5;
            xl_bl_2 = ones(1, obj.r) * 0.5;
            xl_bu_2 = ones(1, obj.r) * 1.5;
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
            f = xu1.^ 2 - 2 * xu1 + xu2 .^ 2 - 2 * xu2 + xl1 .^ 2 + xl2 .^ 2;
         
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
            f = (xl1 - xu1) .^ 2 + (xl2 - xu2) .^ 2;
            
            %-cie
            c = [];
         end
    end
end
