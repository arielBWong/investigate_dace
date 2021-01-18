classdef TNK
    properties
        n_var;
        xl;
        xu;
        n_con;
        n_obj;
        ref;
        name = 'TNK';
        %---
        p = 1;
        q = 2;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
    end
    
    methods
        function obj = TNK()
            obj.n_var = 2;
            obj.xl = [0, 1e-30];
            obj.xu = [pi, pi];
            obj.n_con = 2;
            obj.n_obj = 2;
            obj.ref = [1.2, 1.2];
            obj.name = 'TNK';
            
             %---bl adapt
            obj.n_lvar = obj.q;
            obj.xl_bl =  [0, 1e-30];
            obj.xl_bu =  [pi, pi];
            %-----
        end
        
        function [f, con] = evaluate(obj, x)
            % ----------------------------------------------------------------------------
            %Deb, K.,Pratap, A.,Agarwal, S., et al. A fast and elitist multiobjective
            %genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation
            % 2002, 6 (2): 182-197.
            % x1 = [0,5]; x2 = [0,3].
            % ----------------------------------------------------------------------------
            x1 = x(:,1); x2 = x(:, 2);
            f(:, 1) =x1;
            f(:, 2) =x2;
            con(:, 1) = -x1.^2 - x2.^2 +1 + 0.1*cos(16*atan(x1./x2));
            con(:, 2) = (x1 - 0.5).^2 + (x2 - 0.5).^2 -0.5;
            
        end
        
              function [f, con] = evaluate_l(obj, xu, x)
            % ----------------------------------------------------------------------------
            %Deb, K.,Pratap, A.,Agarwal, S., et al. A fast and elitist multiobjective
            %genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation
            % 2002, 6 (2): 182-197.
            % x1 = [0,5]; x2 = [0,3].
            % ----------------------------------------------------------------------------
            x1 = x(:,1); x2 = x(:, 2);
            f(:, 1) =x1;
            f(:, 2) =x2;
            con(:, 1) = -x1.^2 - x2.^2 +1 + 0.1*cos(16*atan(x1./x2));
            con(:, 2) = (x1 - 0.5).^2 + (x2 - 0.5).^2 -0.5;
            
        end

    end
end