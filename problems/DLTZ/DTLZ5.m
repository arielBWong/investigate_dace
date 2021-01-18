classdef DTLZ5
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
        function obj = DTLZ5(n_var, m, ref)
            if nargin > 0
                obj.n_var = n_var;
                obj.xl = zeros(1, n_var) ;
                obj.xu = ones(1, n_var);
                obj.n_con = 0;
                obj.n_obj = m;
                obj.ref = ones(1, n_obj) * ref;
                obj.name = 'DTLZ5';
            else
                obj.n_var = 6;
                obj.xl = zeros(1, obj.n_var) ;
                obj.xu = ones(1, obj.n_var);
                obj.n_con = 0;
                obj.n_obj = 3;
                obj.ref = ones(1, obj.n_obj) * 2.5;
                obj.name = 'DTLZ5';
            end
        end
        
        function [y, c] = evaluate(obj, x)
            m = obj.n_obj;
           
            g=sum((x(:,m:end)-0.5).^2,2);

            if m>2
                x(:,2:m-1)=(1+2*g(:,ones(1,m-2)).*x(:,2:m-1))./(2*(1+g(:,ones(1,m-2))));
            end
            
            y(:,1)=(1+g).*prod(cos(0.5*pi*x(:,1:m-1)),2);
            if m>2
                for ii=2:m-1
                    y(:,ii)=(1+g).*prod(cos(0.5*pi*x(:,1:m-ii)),2).*sin(0.5*pi*x(:,m-ii+1));
                end
            end
            
            y(:,m)=(1+g).*sin(0.5*pi*x(:,1));
            
            c = [];
        end
    end
end
