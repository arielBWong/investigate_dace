classdef ZDT3
    
    properties
        n_var;
        xl;
        xu;
        n_con;
        n_obj;
        ref;
        name;
    end
    
    methods
        function obj = ZDT3(n_var, m, ref)
            if nargin > 0
                obj.n_var = n_var;
                obj.xl = zeros(1, n_var) ;
                obj.xu = ones(1, n_var);
                obj.n_con = 0;
                obj.n_obj = m;
                obj.ref = ones(1, n_obj) * ref;
                obj.name = 'ZDT3';
            else
                obj.n_var = 6;
                obj.xl = zeros(1, obj.n_var) ;
                obj.xu = ones(1, obj.n_var);
                obj.n_con = 0;
                obj.n_obj = 2;
                obj.ref = ones(1, obj.n_obj) * 1.1;
                obj.name = 'ZDT3';
            end
        end
        
        function [y, c] = evaluate(obj, x)
            
            % function only for test, incomplete implementation
                       
            y(:,1)=x(:,1);
            g=1+(9/(size(x,2)-1))*sum(x(:,2:end),2);
            h=1-(y(:,1)./g).^0.5-(y(:,1)./g).*sin(10*pi*x(:,1));
            y(:,2)=g.*h;
            
            c = [];
        end
    end
end