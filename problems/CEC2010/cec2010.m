classdef cec2010
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
        cfunc_num;
        name;
        uopt = NaN;
        lopt = NaN; % double check needed
    end
    methods
        function obj = cec2010(cfunc_num, dim)
            % level variables       
            obj.name = strcat('cec2010 ', num2str(cfunc_num));
            obj.cfunc_num = cfunc_num;
            obj.n_lvar = dim;
            obj.n_uvar = dim;

            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);        
             
            
           
            % init bound lower level
            [obj.xl_bl, obj.xl_bu] = get_bound(dim, cfunc_num);
            [~, ~, ~, obj. xprime] = cec10_cop([],cfunc_num, true); 
            obj.xprime = obj.xprime(1:dim);
            
        end
        
        function [f, c] = evaluate_u(obj, xu, xl) 
            %-obj
            f = [];
            c = [];
            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
            
             [f, g, h, ~] = cec10_cop(xl,obj.cfunc_num);          
             c = [g, h-(1e-3)];
         end
    end
end
