classdef tp2
    properties
        p = 1;
        q ;
        n_lvar;
        n_uvar;
        xu_bl;
        xu_bu;
        xl_bl;
        xl_bu;
        name = 'tp2';
        uopt = NaN;
        lopt = NaN; % double check needed
    end
    methods
        function obj = tp2(k)
            obj.q = k;
            % level variables
            obj.n_lvar = obj.q;
            obj.n_uvar = obj.p;
            
            % bounds
            %init bound upper level
            obj.xu_bl = ones(1, obj.p) * -1;
            obj.xu_bu = ones(1, obj.p) * 2;
           
           
            % init bound lower level
            obj.xl_bl = ones(1, obj.q) * -1;
            obj.xl_bu = ones(1, obj.q) * 2;      
        end
        
        function [f, c] = evaluate_u(obj, xu, xl) 
            %-obj
           f(:, 1) = (xl(:, 1) - 1) .^ 2 + sum(xl(:, 2: obj.q) .^ 2, 2) + xu.^2;
           f(:, 2) = (xl(:, 1) - 1) .^ 2 +  sum(xl(:, 2: obj.q) .^ 2, 2) + (xu - 1).^2;
          
                 
            %-cie
            c= [];            
        end
        
        
        function [f, c] = evaluate_l(obj, xu, xl)          
            %-obj
            f(:, 1) =  xl(:, 1) .^ 2 + sum(xl(:, 2: obj.q) .^ 2, 2) ;
            f(:, 2) =  (xl(:, 1) - xu) .^ 2 + sum(xl(:, 2: obj.q) .^ 2, 2);
            f = sum(f, 2);
            
            %-cie
           c=[];
        
        end
         
            
            % make sure folder evenpf is placed under main project root
%             workdir = pwd;
%             idcs = strfind(workdir, '\');
%             upperfolder = workdir(1: idcs(end)-1);
%             
%             idcs = strfind(upperfolder, '\');
%             uupfolder = upperfolder(1: idcs(end)-1);
%             
%             method_folder = strcat(uupfolder,'\ND_Sort');
%             addpath(method_folder);
% 
%             [id_fronts,f_fronts,~,~] = E_NDSort_c(PF);
%             updated_order=Sparse_selection(id_fronts,f_fronts,num_point);
%             pf = PF(updated_order(1:num_point), :);
           
         
    end
end
