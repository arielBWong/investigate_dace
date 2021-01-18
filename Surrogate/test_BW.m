function f=test_BW(xl)
b = [0.05, 0.2, 0.3, 0.3]';
C = [-4, 4, -2, 2;
    -4, 4, -2, 2;
    -4, 4, -2, 2;
    -4, 4, -2, 2];
outer = 0;
nv = 1;
for ii = 1:4
    bi = b(ii);
    inner = 0;
    for jj = 1:nv
        xj = xl(:, jj);
        Cji = C(jj, ii);
        inner = inner + (xj-Cji).^2;
    end
    outer = outer + 1./(inner+bi);
end
outer = outer - sum(xl.^2, 2);
f = -outer;
c = [];
return
