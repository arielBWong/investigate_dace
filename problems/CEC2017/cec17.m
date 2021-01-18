classdef cec17
    properties
        p = 2;
        q = 2;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        xprime = [];
        cfunc_num;
        name;
        uopt = NaN;
        lopt = NaN; % double check needed
    end
    methods
        function obj = cec17(cfunc_num, dim)
            % level variables       
            obj.name = strcat('cec2017 ', num2str(cfunc_num));
            obj.cfunc_num = cfunc_num;
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;

            %init bound upper level
            obj.xu_bl = zeros(1, obj.p);
            obj.xu_bu = ones(1, obj.p);        
             
            
           
            % init bound lower level
            [obj.xl_bl, obj.xl_bu] = get_bound17(dim, cfunc_num);

        end
        
        function [f, c] = evaluate_u(obj, xu, xl) 
            %-obj
            f = [];
            c = [];
            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)
            
            [f,g,h] = cec2017(xl,obj.cfunc_num);
            c = [g, h-(1e-4)];
         end
    end
end
