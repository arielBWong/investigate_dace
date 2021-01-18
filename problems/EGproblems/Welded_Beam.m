classdef Welded_Beam
    properties
        n_var;
        xl;
        xu;
        n_con;
        n_obj;
        ref;
        name = 'Welded_Beam';
        %----
        p = 1;
        q = 4;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
    end
    
    methods
        function obj = Welded_Beam()
            obj.n_var = 4;
            obj.xl = [0.125, 0.1, 0.1, 0.125];
            obj.xu = [5, 10, 10, 5];
            
            %---bl adapt
            obj.n_lvar = obj.q;
            obj.xl_bl =  [0.125, 0.1, 0.1, 0.125];
            obj.xl_bu =  [5, 10, 10, 5];
            %-----
            
            obj.n_con = 4;
            obj.n_obj = 2;
            obj.ref = [100, 0.08];
            obj.name = 'Welded_Beam';
        end
        
        function [f, con] = evaluate(obj, x)
            % ----------------------------------------------------------------------------
            % Deb, K. Multi-objective optimization using evolutionary algorithms.
            % John Wiley & Sons, 2001.
            % x1 = [0125,10], x2,x3,x4 = [0.1,10]
            % ----------------------------------------------------------------------------
            h = x(:,1); l = x(:, 2); t = x(:, 3); b = x(:, 4);
            
            tao_1 = 6000./(sqrt(2)*h.*l);
            tao_2 = 6000*(14 + 0.5*l).*sqrt(0.25*(l.^2 + (h+t).^2))./(2*(0.707*h.*l.*(l.^2/12 + 0.25*(h+t).^2)));
            sigma = 504000./(t.^2.*b);
            Pc = 64746.022.*(1-0.0282346*t).*t.*b.^3;
            tao = sqrt(tao_1.^2 + tao_2.^2 + l.*tao_1.*tao_2./(sqrt(0.25*(l.^2 + (h+t).^2))));
            
            f(:,1)=1.10471*h.^2.*l + 0.04811*t.*b.*(14+l);
            f(:,2)=2.1952./(t.^3.*b);
            
            con(:,1)=tao  - 13600;
            con(:,2)=sigma - 30000;
            con(:,3)=h - b;
            con(:,4)=6000 - Pc;
        end
        
        function [f, con] = evaluate_l(obj, xu, x)
            % ----------------------------------------------------------------------------
            % Deb, K. Multi-objective optimization using evolutionary algorithms.
            % John Wiley & Sons, 2001.
            % x1 = [0125,10], x2,x3,x4 = [0.1,10]
            % ----------------------------------------------------------------------------
            h = x(:,1); l = x(:, 2); t = x(:, 3); b = x(:, 4);
            
            tao_1 = 6000./(sqrt(2)*h.*l);
            tao_2 = 6000*(14 + 0.5*l).*sqrt(0.25*(l.^2 + (h+t).^2))./(2*(0.707*h.*l.*(l.^2/12 + 0.25*(h+t).^2)));
            sigma = 504000./(t.^2.*b);
            Pc = 64746.022.*(1-0.0282346*t).*t.*b.^3;
            tao = sqrt(tao_1.^2 + tao_2.^2 + l.*tao_1.*tao_2./(sqrt(0.25*(l.^2 + (h+t).^2))));
            
            f(:,1)=1.10471*h.^2.*l + 0.04811*t.*b.*(14+l);
            f(:,2)=2.1952./(t.^3.*b);
            
            con(:,1)= tao  - 13600;
            con(:,2)= sigma - 30000;
            con(:,3)=h - b;
            con(:,4)=6000 - Pc;
        end
    end
end

