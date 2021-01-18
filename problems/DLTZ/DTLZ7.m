classdef DTLZ7
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
        function obj = DTLZ7(n_var, m, ref)
            if nargin > 0
                obj.n_var = n_var;
                obj.xl = zeros(1, n_var) ;
                obj.xu = ones(1, n_var);
                obj.n_con = 0;
                obj.n_obj = m;
                obj.ref = ones(1, n_obj) * ref;
                obj.name = 'DTLZ7';
            else
                obj.n_var = 6;
                obj.xl = zeros(1, obj.n_var) ;
                obj.xu = ones(1, obj.n_var);
                obj.n_con = 0;
                obj.n_obj = 3;
                obj.ref = ones(1, obj.n_obj) * 30;
                obj.name = 'DTLZ7';
            end
        end
        
        function [y, c] = evaluate(obj, x)
            m = obj.n_obj;
            n=size(x,2);     %number of design variable
            k=n-m+1;
            g=1+(9/k)*sum(x(:,m:end),2);
            y(:,1:m-1)=x(:,1:m-1);
            h=m-sum((y(:,1:m-1)./(1+g(:,ones(1,m-1)))).*(1+sin(3*pi*y(:,1:m-1))),2);
            y(:,m)=(1+g).*h;
            
            
            c = [];
        end
    end
end
