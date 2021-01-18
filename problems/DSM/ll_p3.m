function p3 = ll_p3(xu, xl)
n_lvar = size(xl, 2);
n_uvar = size(xu, 2);
if n_lvar <=3
    k = 10;
elseif n_lvar == 4
    k = 20;
else
    k = 30;
end
p3 = k * abs(sin(pi/n_lvar.* (xl(:, 2:n_lvar) - xu(:, 2:n_uvar))));
end